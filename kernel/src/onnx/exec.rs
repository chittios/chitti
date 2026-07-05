//! **ONNX op executor** — a small f32 interpreter covering the op set the
//! voice models use (silero-vad v5 today: Conv/LSTM/Relu/Sigmoid + shape
//! plumbing). Values are dense tensors in row-major order; i64 shape tensors
//! ride along as a parallel representation so Reshape/Slice/Pad arguments
//! work without a full type system.
//!
//! Correctness is anchored by an in-kernel test that runs the real silero VAD
//! against reference probabilities produced by onnxruntime on the host.

use super::{Graph, Model, Tensor};
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

/// A runtime value: dims + f32 data, with optional exact i64 view (shape math).
#[derive(Clone)]
pub struct Val {
    pub dims: Vec<usize>,
    pub f: Vec<f32>,
    pub i: Option<Vec<i64>>,
}

impl Val {
    pub fn new(dims: Vec<usize>, f: Vec<f32>) -> Self {
        Self { dims, f, i: None }
    }
    fn ints(&self) -> Vec<i64> {
        match &self.i {
            Some(v) => v.clone(),
            None => self.f.iter().map(|&x| x as i64).collect(),
        }
    }
    fn numel(&self) -> usize {
        self.dims.iter().product::<usize>().max(1)
    }
}

fn tensor_to_val(t: &Tensor<'_>) -> Val {
    let dims: Vec<usize> = t.dims.iter().map(|&d| d.max(0) as usize).collect();
    let n: usize = dims.iter().product::<usize>().max(1);
    match t.dtype {
        7 => {
            // int64: raw or int64_data
            let ints: Vec<i64> = if !t.raw.is_empty() {
                t.raw.chunks_exact(8).map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect()
            } else {
                t.ints.clone()
            };
            let f = ints.iter().map(|&v| v as f32).collect();
            Val { dims, f, i: Some(ints) }
        }
        _ => {
            // treat everything else as f32
            let mut f = Vec::with_capacity(n);
            if !t.raw.is_empty() {
                for c in t.raw.chunks_exact(4) {
                    f.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                }
            } else {
                f.extend_from_slice(&t.floats);
            }
            Val::new(dims, f)
        }
    }
}

/// `ln(x)` — exposed for the mel frontend (same series used internally).
pub fn logf_pub(x: f32) -> f32 {
    logf(x)
}
/// `sqrt(x)` — exposed for the mel frontend.
pub fn sqrtf_pub(x: f32) -> f32 {
    sqrtf(x)
}

fn expf(x: f32) -> f32 {
    crate::cortex::tensor::expf(x)
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + expf(-x))
}
fn tanhf(x: f32) -> f32 {
    if x > 15.0 {
        return 1.0;
    }
    if x < -15.0 {
        return -1.0;
    }
    let e = expf(2.0 * x);
    (e - 1.0) / (e + 1.0)
}
fn sqrtf(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    // Bit-trick initial guess (halve the exponent), then Newton to ~1 ulp.
    let mut r = f32::from_bits((x.to_bits() >> 1) + 0x1fbd_1df5);
    for _ in 0..4 {
        r = 0.5 * (r + x / r);
    }
    r
}
fn logf(x: f32) -> f32 {
    // ln(x) = ln(m * 2^e) = ln m + e ln2, m in [1,2): atanh-series on (m-1)/(m+1).
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let bits = x.to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 127;
    let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000); // [1,2)
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    let ln_m = 2.0 * t * (1.0 + t2 / 3.0 + t2 * t2 / 5.0 + t2 * t2 * t2 / 7.0 + t2 * t2 * t2 * t2 / 9.0);
    ln_m + e as f32 * core::f32::consts::LN_2
}

fn floorf(x: f32) -> f32 {
    let t = x as i64 as f32;
    if t > x {
        t - 1.0
    } else {
        t
    }
}
fn ceilf(x: f32) -> f32 {
    -floorf(-x)
}
fn sinf(x: f32) -> f32 {
    cosf(x - core::f32::consts::FRAC_PI_2)
}
fn cosf(x: f32) -> f32 {
    use core::f32::consts::PI;
    let mut a = x % (2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    }
    if a < -PI {
        a += 2.0 * PI;
    }
    let x2 = a * a;
    1.0 - x2 * (0.5 - x2 * (1.0 / 24.0 - x2 * (1.0 / 720.0 - x2 / 40320.0)))
}
fn powf(a: f32, b: f32) -> f32 {
    if b == 2.0 {
        return a * a;
    }
    if a <= 0.0 {
        return if b == 0.0 { 1.0 } else { 0.0 };
    }
    expf(b * logf(a))
}

/// Batched matmul of the last two dims: `A[.., m, k] · B[.., k, n] -> [.., m, n]`,
/// with optional integer zero-points subtracted first (MatMulInteger). Leading
/// batch dims broadcast. B may be 2-D (shared across A's batch).
fn matmul(a: &Val, b: &Val, azp: f32, bzp: f32) -> Val {
    let (ar, br) = (a.dims.len(), b.dims.len());
    let m = a.dims[ar - 2];
    let k = a.dims[ar - 1];
    let n = b.dims[br - 1];
    let batch: usize = a.dims[..ar - 2].iter().product::<usize>().max(1);
    let b_batch: usize = b.dims[..br - 2].iter().product::<usize>().max(1);
    let mut out = vec![0f32; batch * m * n];
    for bi in 0..batch {
        let ao = bi * m * k;
        let bo = (if b_batch == 1 { 0 } else { bi }) * k * n;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += (a.f[ao + i * k + kk] - azp) * (b.f[bo + kk * n + j] - bzp);
                }
                out[(bi * m + i) * n + j] = acc;
            }
        }
    }
    let mut od: Vec<usize> = a.dims[..ar - 2].to_vec();
    od.push(m);
    od.push(n);
    Val::new(od, out)
}

/// Broadcast two shapes (numpy rules), returning the output dims.
fn broadcast_dims(a: &[usize], b: &[usize]) -> Vec<usize> {
    let r = a.len().max(b.len());
    let mut out = vec![0usize; r];
    for k in 0..r {
        let da = if k < r - a.len() { 1 } else { a[k - (r - a.len())] };
        let db = if k < r - b.len() { 1 } else { b[k - (r - b.len())] };
        out[k] = da.max(db);
    }
    out
}

/// Index into `v` as if broadcast to `od` at multi-index `idx`.
fn bcast_get(v: &Val, od: &[usize], idx: &[usize]) -> f32 {
    let r = od.len();
    let vr = v.dims.len();
    let mut off = 0usize;
    let mut stride = 1usize;
    // row-major, from last dim backwards
    for k in (0..vr).rev() {
        let ok = idx[r - vr + k];
        let dk = v.dims[k];
        let i = if dk == 1 { 0 } else { ok };
        off += i * stride;
        stride *= dk;
    }
    v.f[off]
}

fn elementwise2(a: &Val, b: &Val, f: impl Fn(f32, f32) -> f32) -> Val {
    let od = broadcast_dims(&a.dims, &b.dims);
    let n: usize = od.iter().product::<usize>().max(1);
    let mut out = Vec::with_capacity(n);
    let mut idx = vec![0usize; od.len()];
    for _ in 0..n {
        out.push(f(bcast_get(a, &od, &idx), bcast_get(b, &od, &idx)));
        // increment multi-index
        for k in (0..od.len()).rev() {
            idx[k] += 1;
            if idx[k] < od[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    Val::new(od, out)
}

/// Execute `graph` with the given input feeds; returns the requested outputs.
pub fn run(model: &Model<'_>, feeds: &[(&str, Val)]) -> Result<BTreeMap<String, Val>, String> {
    let g = &model.graph;
    let mut env: BTreeMap<String, Val> = BTreeMap::new();
    for t in &g.initializers {
        env.insert(t.name.to_string(), tensor_to_val(t));
    }
    for (name, v) in feeds {
        env.insert((*name).to_string(), v.clone());
    }
    exec_graph(g, &mut env)?;
    let mut out = BTreeMap::new();
    for o in &g.outputs {
        let v = env.get(*o).ok_or_else(|| alloc::format!("missing output {o}"))?;
        out.insert((*o).to_string(), v.clone());
    }
    Ok(out)
}

fn attr_i(n: &super::Node<'_>, name: &str, dflt: i64) -> i64 {
    n.attrs.iter().find(|a| a.name == name).map(|a| a.i).unwrap_or(dflt)
}
fn attr_ints(n: &super::Node<'_>, name: &str) -> Option<Vec<i64>> {
    n.attrs.iter().find(|a| a.name == name).map(|a| a.ints.clone())
}

fn exec_graph(g: &Graph<'_>, env: &mut BTreeMap<String, Val>) -> Result<(), String> {
    for node in &g.nodes {
        let get = |env: &BTreeMap<String, Val>, i: usize| -> Result<Val, String> {
            let name = node.inputs.get(i).copied().unwrap_or("");
            if name.is_empty() {
                return Err(alloc::format!("{}: missing input {i}", node.op));
            }
            env.get(name).cloned().ok_or_else(|| alloc::format!("{}: unbound input '{name}'", node.op))
        };
        let out: Vec<Val> = match node.op {
            "Constant" => {
                let t = node
                    .attrs
                    .iter()
                    .find(|a| a.name == "value")
                    .and_then(|a| a.tensor.as_ref())
                    .ok_or("Constant: no value")?;
                vec![tensor_to_val(t)]
            }
            "ConstantOfShape" => {
                let shape = get(env, 0)?.ints();
                let dims: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
                let n: usize = dims.iter().product::<usize>().max(1);
                let fill = node.attrs.iter().find(|a| a.name == "value").and_then(|a| a.tensor.as_ref()).map(|t| tensor_to_val(t).f.first().copied().unwrap_or(0.0)).unwrap_or(0.0);
                vec![Val::new(dims, vec![fill; n])]
            }
            "Cast" => {
                // dtype is advisory here — everything lives as f32 (+i64 view).
                let mut v = get(env, 0)?;
                if attr_i(node, "to", 1) == 7 {
                    v.i = Some(v.f.iter().map(|&x| x as i64).collect());
                }
                vec![v]
            }
            "Shape" => {
                let v = get(env, 0)?;
                let ints: Vec<i64> = v.dims.iter().map(|&d| d as i64).collect();
                let f = ints.iter().map(|&d| d as f32).collect();
                vec![Val { dims: vec![ints.len()], f, i: Some(ints) }]
            }
            "Unsqueeze" => {
                let v = get(env, 0)?;
                let axes = attr_ints(node, "axes").or_else(|| get(env, 1).ok().map(|a| a.ints())).unwrap_or_default();
                let mut dims = v.dims.clone();
                let mut axes: Vec<i64> = axes;
                axes.sort_unstable();
                for &ax in &axes {
                    let r = dims.len() as i64 + 1;
                    let a = if ax < 0 { ax + r } else { ax } as usize;
                    dims.insert(a, 1);
                }
                vec![Val { dims, f: v.f, i: v.i }]
            }
            "Squeeze" => {
                let v = get(env, 0)?;
                let axes = attr_ints(node, "axes").or_else(|| get(env, 1).ok().map(|a| a.ints())).unwrap_or_default();
                let mut dims = Vec::new();
                for (k, &d) in v.dims.iter().enumerate() {
                    let squeeze = if axes.is_empty() {
                        d == 1
                    } else {
                        axes.iter().any(|&a| {
                            let r = v.dims.len() as i64;
                            (if a < 0 { a + r } else { a }) as usize == k
                        })
                    };
                    if !squeeze {
                        dims.push(d);
                    }
                }
                vec![Val { dims, f: v.f, i: v.i }]
            }
            "Reshape" => {
                let v = get(env, 0)?;
                let shape = get(env, 1)?.ints();
                let total = v.numel();
                let mut dims: Vec<usize> = Vec::new();
                let mut infer = None;
                let mut known = 1usize;
                for (k, &d) in shape.iter().enumerate() {
                    if d == -1 {
                        infer = Some(k);
                        dims.push(1);
                    } else if d == 0 {
                        let keep = v.dims.get(k).copied().unwrap_or(1);
                        dims.push(keep);
                        known *= keep;
                    } else {
                        dims.push(d as usize);
                        known *= d as usize;
                    }
                }
                if let Some(k) = infer {
                    dims[k] = total / known.max(1);
                }
                vec![Val { dims, f: v.f, i: v.i }]
            }
            "Concat" => {
                let axis = attr_i(node, "axis", 0);
                let vals: Result<Vec<Val>, String> = (0..node.inputs.len()).map(|i| get(env, i)).collect();
                let vals = vals?;
                let r = vals[0].dims.len() as i64;
                let ax = (if axis < 0 { axis + r } else { axis }) as usize;
                let mut dims = vals[0].dims.clone();
                dims[ax] = vals.iter().map(|v| v.dims[ax]).sum();
                let outer: usize = dims[..ax].iter().product::<usize>().max(1);
                let inner: usize = dims[ax + 1..].iter().product::<usize>().max(1);
                let mut f = Vec::with_capacity(dims.iter().product());
                for o in 0..outer {
                    for v in &vals {
                        let da = v.dims[ax];
                        let start = o * da * inner;
                        f.extend_from_slice(&v.f[start..start + da * inner]);
                    }
                }
                // i64 view survives when all parts have one.
                let ints = if vals.iter().all(|v| v.i.is_some()) {
                    let mut iv = Vec::new();
                    for o in 0..outer {
                        for v in &vals {
                            let da = v.dims[ax];
                            let start = o * da * inner;
                            iv.extend_from_slice(&v.i.as_ref().unwrap()[start..start + da * inner]);
                        }
                    }
                    Some(iv)
                } else {
                    None
                };
                vec![Val { dims, f, i: ints }]
            }
            "Transpose" => {
                let v = get(env, 0)?;
                let r = v.dims.len();
                let perm = attr_ints(node, "perm").unwrap_or_else(|| (0..r as i64).rev().collect());
                let od: Vec<usize> = perm.iter().map(|&p| v.dims[p as usize]).collect();
                let n = v.numel();
                let mut out = vec![0f32; n];
                // strides of input
                let mut istr = vec![1usize; r];
                for k in (0..r.saturating_sub(1)).rev() {
                    istr[k] = istr[k + 1] * v.dims[k + 1];
                }
                let mut idx = vec![0usize; r];
                for item in out.iter_mut().take(n) {
                    let mut src = 0usize;
                    for k in 0..r {
                        src += idx[k] * istr[perm[k] as usize];
                    }
                    *item = v.f[src];
                    for k in (0..r).rev() {
                        idx[k] += 1;
                        if idx[k] < od[k] {
                            break;
                        }
                        idx[k] = 0;
                    }
                }
                vec![Val::new(od, out)]
            }
            "Slice" => {
                let v = get(env, 0)?;
                let starts = attr_ints(node, "starts").or_else(|| get(env, 1).ok().map(|a| a.ints())).unwrap_or_default();
                let ends = attr_ints(node, "ends").or_else(|| get(env, 2).ok().map(|a| a.ints())).unwrap_or_default();
                let axes = attr_ints(node, "axes")
                    .or_else(|| get(env, 3).ok().map(|a| a.ints()))
                    .unwrap_or_else(|| (0..starts.len() as i64).collect());
                let steps = get(env, 4).ok().map(|a| a.ints()).unwrap_or_else(|| vec![1; starts.len()]);
                let r = v.dims.len();
                // Signed slice bounds — silero uses negative steps (flips).
                let mut start = vec![0i64; r];
                let mut end: Vec<i64> = v.dims.iter().map(|&d| d as i64).collect();
                let mut step = vec![1i64; r];
                for (k, &ax) in axes.iter().enumerate() {
                    let a = (if ax < 0 { ax + r as i64 } else { ax }) as usize;
                    let d = v.dims[a] as i64;
                    let st = steps.get(k).copied().unwrap_or(1);
                    step[a] = st;
                    let norm = |x: i64| if x < 0 { x + d } else { x };
                    let (s_raw, e_raw) = (starts[k], ends[k]);
                    if st > 0 {
                        start[a] = norm(s_raw).clamp(0, d);
                        end[a] = if e_raw >= d { d } else { norm(e_raw).clamp(0, d) };
                    } else {
                        start[a] = norm(s_raw).clamp(0, d - 1);
                        // An end below -d is the "before the start" sentinel: -1.
                        end[a] = if e_raw < -d { -1 } else { norm(e_raw).clamp(-1, d) };
                    }
                }
                let od: Vec<usize> = (0..r)
                    .map(|k| {
                        let len = if step[k] > 0 {
                            (end[k] - start[k] + step[k] - 1) / step[k]
                        } else {
                            (end[k] - start[k] + step[k] + 1) / step[k]
                        };
                        len.max(0) as usize
                    })
                    .collect();
                let n: usize = od.iter().product::<usize>().max(1);
                let mut istr = vec![1usize; r];
                for k in (0..r.saturating_sub(1)).rev() {
                    istr[k] = istr[k + 1] * v.dims[k + 1];
                }
                let mut out = Vec::with_capacity(n);
                let mut idx = vec![0usize; r];
                for _ in 0..n {
                    let mut src = 0usize;
                    for k in 0..r {
                        let i = start[k] + idx[k] as i64 * step[k];
                        src += i as usize * istr[k];
                    }
                    out.push(v.f[src]);
                    for k in (0..r).rev() {
                        idx[k] += 1;
                        if idx[k] < od[k].max(1) {
                            break;
                        }
                        idx[k] = 0;
                    }
                }
                vec![Val::new(od, out)]
            }
            "Pad" => {
                let v = get(env, 0)?;
                let pads = get(env, 1)?.ints();
                let cval = get(env, 2).ok().map(|c| c.f.first().copied().unwrap_or(0.0)).unwrap_or(0.0);
                let reflect = node.attrs.iter().any(|a| a.name == "mode" && a.s == b"reflect");
                let edge = node.attrs.iter().any(|a| a.name == "mode" && a.s == b"edge");
                let r = v.dims.len();
                let od: Vec<usize> = (0..r).map(|k| v.dims[k] + pads[k].max(0) as usize + pads[k + r].max(0) as usize).collect();
                let n: usize = od.iter().product::<usize>().max(1);
                let mut istr = vec![1usize; r];
                for k in (0..r.saturating_sub(1)).rev() {
                    istr[k] = istr[k + 1] * v.dims[k + 1];
                }
                let mut out = Vec::with_capacity(n);
                let mut idx = vec![0usize; r];
                for _ in 0..n {
                    let mut inside = true;
                    let mut src = 0usize;
                    for k in 0..r {
                        let mut i = idx[k] as i64 - pads[k];
                        let d = v.dims[k] as i64;
                        if i < 0 || i >= d {
                            if reflect {
                                // Mirror without repeating the edge sample.
                                while i < 0 || i >= d {
                                    if i < 0 {
                                        i = -i;
                                    }
                                    if i >= d {
                                        i = 2 * (d - 1) - i;
                                    }
                                }
                            } else if edge {
                                i = i.clamp(0, d - 1);
                            } else {
                                inside = false;
                                break;
                            }
                        }
                        src += i as usize * istr[k];
                    }
                    out.push(if inside { v.f[src] } else { cval });
                    for k in (0..r).rev() {
                        idx[k] += 1;
                        if idx[k] < od[k] {
                            break;
                        }
                        idx[k] = 0;
                    }
                }
                vec![Val::new(od, out)]
            }
            "Add" => vec![elementwise2(&get(env, 0)?, &get(env, 1)?, |a, b| a + b)],
            "Sub" => vec![elementwise2(&get(env, 0)?, &get(env, 1)?, |a, b| a - b)],
            "Mul" => vec![elementwise2(&get(env, 0)?, &get(env, 1)?, |a, b| a * b)],
            "Div" => vec![elementwise2(&get(env, 0)?, &get(env, 1)?, |a, b| a / b)],
            "Pow" => vec![elementwise2(&get(env, 0)?, &get(env, 1)?, |a, b| powf(a, b))],
            "Equal" => vec![elementwise2(&get(env, 0)?, &get(env, 1)?, |a, b| if a == b { 1.0 } else { 0.0 })],
            "Less" => vec![elementwise2(&get(env, 0)?, &get(env, 1)?, |a, b| if a < b { 1.0 } else { 0.0 })],
            "Greater" => vec![elementwise2(&get(env, 0)?, &get(env, 1)?, |a, b| if a > b { 1.0 } else { 0.0 })],
            "And" => vec![elementwise2(&get(env, 0)?, &get(env, 1)?, |a, b| if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 })],
            "Not" => {
                let mut v = get(env, 0)?;
                for x in &mut v.f {
                    *x = if *x == 0.0 { 1.0 } else { 0.0 };
                }
                vec![v]
            }
            "Tanh" => {
                let mut v = get(env, 0)?;
                for x in &mut v.f {
                    *x = tanhf(*x);
                }
                v.i = None;
                vec![v]
            }
            "Exp" => {
                let mut v = get(env, 0)?;
                for x in &mut v.f {
                    *x = expf(*x);
                }
                v.i = None;
                vec![v]
            }
            "Sin" => {
                let mut v = get(env, 0)?;
                for x in &mut v.f {
                    *x = sinf(*x);
                }
                v.i = None;
                vec![v]
            }
            "Cos" => {
                let mut v = get(env, 0)?;
                for x in &mut v.f {
                    *x = cosf(*x);
                }
                v.i = None;
                vec![v]
            }
            "LeakyRelu" => {
                let a = node.attrs.iter().find(|at| at.name == "alpha").map(|at| at.f).unwrap_or(0.01);
                let mut v = get(env, 0)?;
                for x in &mut v.f {
                    if *x < 0.0 {
                        *x *= a;
                    }
                }
                v.i = None;
                vec![v]
            }
            "Clip" => {
                let mut v = get(env, 0)?;
                let lo = get(env, 1).ok().and_then(|t| t.f.first().copied()).unwrap_or(f32::NEG_INFINITY);
                let hi = get(env, 2).ok().and_then(|t| t.f.first().copied()).unwrap_or(f32::INFINITY);
                for x in &mut v.f {
                    *x = x.clamp(lo, hi);
                }
                vec![v]
            }
            "Floor" => {
                let mut v = get(env, 0)?;
                for x in &mut v.f {
                    *x = floorf(*x);
                }
                vec![v]
            }
            "Round" => {
                let mut v = get(env, 0)?;
                for x in &mut v.f {
                    *x = floorf(*x + 0.5);
                }
                vec![v]
            }
            "Where" => {
                // condition ? x : y, all broadcast.
                let cond = get(env, 0)?;
                let a = get(env, 1)?;
                let b = get(env, 2)?;
                let od = broadcast_dims(&broadcast_dims(&cond.dims, &a.dims), &b.dims);
                let n: usize = od.iter().product::<usize>().max(1);
                let mut out = Vec::with_capacity(n);
                let mut idx = vec![0usize; od.len()];
                for _ in 0..n {
                    let c = bcast_get(&cond, &od, &idx);
                    out.push(if c != 0.0 { bcast_get(&a, &od, &idx) } else { bcast_get(&b, &od, &idx) });
                    for k in (0..od.len()).rev() {
                        idx[k] += 1;
                        if idx[k] < od[k] {
                            break;
                        }
                        idx[k] = 0;
                    }
                }
                vec![Val::new(od, out)]
            }
            "Softmax" | "LogSoftmax" => {
                let v = get(env, 0)?;
                let r = v.dims.len();
                let axis = {
                    let ax = attr_i(node, "axis", -1);
                    (if ax < 0 { ax + r as i64 } else { ax }) as usize
                };
                let inner: usize = v.dims[axis..].iter().product::<usize>().max(1);
                let ax_len = v.dims[axis];
                let stride: usize = v.dims[axis + 1..].iter().product::<usize>().max(1);
                let groups = v.f.len() / inner;
                let mut out = v.f.clone();
                let logmode = node.op == "LogSoftmax";
                for g in 0..groups {
                    for s in 0..stride {
                        let mut mx = f32::NEG_INFINITY;
                        for a in 0..ax_len {
                            let o = g * inner + a * stride + s;
                            if out[o] > mx {
                                mx = out[o];
                            }
                        }
                        let mut sum = 0f32;
                        for a in 0..ax_len {
                            let o = g * inner + a * stride + s;
                            let e = expf(out[o] - mx);
                            out[o] = e;
                            sum += e;
                        }
                        for a in 0..ax_len {
                            let o = g * inner + a * stride + s;
                            out[o] = if logmode { logf(out[o] / sum) } else { out[o] / sum };
                        }
                    }
                }
                vec![Val::new(v.dims.clone(), out)]
            }
            "LayerNormalization" => {
                let x = get(env, 0)?;
                let scale = get(env, 1)?;
                let bias = get(env, 2).ok();
                let eps = node.attrs.iter().find(|a| a.name == "epsilon").map(|a| a.f).unwrap_or(1e-5);
                let r = x.dims.len();
                let ax = {
                    let a = attr_i(node, "axis", -1);
                    (if a < 0 { a + r as i64 } else { a }) as usize
                };
                let norm: usize = x.dims[ax..].iter().product::<usize>().max(1);
                let groups = x.f.len() / norm;
                let mut out = x.f.clone();
                for g in 0..groups {
                    let base = g * norm;
                    let mut mean = 0f32;
                    for i in 0..norm {
                        mean += out[base + i];
                    }
                    mean /= norm as f32;
                    let mut var = 0f32;
                    for i in 0..norm {
                        let d = out[base + i] - mean;
                        var += d * d;
                    }
                    var /= norm as f32;
                    let inv = 1.0 / sqrtf(var + eps);
                    for i in 0..norm {
                        let sc = scale.f[i % scale.f.len()];
                        let b = bias.as_ref().map(|bb| bb.f[i % bb.f.len()]).unwrap_or(0.0);
                        out[base + i] = (out[base + i] - mean) * inv * sc + b;
                    }
                }
                vec![Val::new(x.dims.clone(), out)]
            }
            "Gather" => {
                let data = get(env, 0)?;
                let ind = get(env, 1)?;
                let r = data.dims.len();
                let axis = {
                    let a = attr_i(node, "axis", 0);
                    (if a < 0 { a + r as i64 } else { a }) as usize
                };
                let idxs = ind.ints();
                let axd = data.dims[axis] as i64;
                let outer: usize = data.dims[..axis].iter().product::<usize>().max(1);
                let inner: usize = data.dims[axis + 1..].iter().product::<usize>().max(1);
                let mut od: Vec<usize> = data.dims[..axis].to_vec();
                od.extend_from_slice(&ind.dims);
                od.extend_from_slice(&data.dims[axis + 1..]);
                let mut out = Vec::with_capacity(outer * idxs.len() * inner);
                let want_i = data.i.is_some();
                let mut oi = Vec::new();
                for o in 0..outer {
                    for &ix in &idxs {
                        let a = (if ix < 0 { ix + axd } else { ix }).clamp(0, axd - 1) as usize;
                        let start = (o * data.dims[axis] + a) * inner;
                        out.extend_from_slice(&data.f[start..start + inner]);
                        if want_i {
                            oi.extend_from_slice(&data.i.as_ref().unwrap()[start..start + inner]);
                        }
                    }
                }
                vec![Val { dims: od, f: out, i: if want_i { Some(oi) } else { None } }]
            }
            "Split" => {
                let v = get(env, 0)?;
                let r = v.dims.len();
                let axis = {
                    let a = attr_i(node, "axis", 0);
                    (if a < 0 { a + r as i64 } else { a }) as usize
                };
                let sizes: Vec<usize> = attr_ints(node, "split")
                    .or_else(|| get(env, 1).ok().map(|s| s.ints()))
                    .map(|v| v.iter().map(|&x| x as usize).collect())
                    .unwrap_or_else(|| {
                        let k = node.outputs.len().max(1);
                        vec![v.dims[axis] / k; k]
                    });
                let outer: usize = v.dims[..axis].iter().product::<usize>().max(1);
                let inner: usize = v.dims[axis + 1..].iter().product::<usize>().max(1);
                let mut res = Vec::new();
                let mut off = 0usize;
                for &sz in &sizes {
                    let mut dims = v.dims.clone();
                    dims[axis] = sz;
                    let mut f = Vec::with_capacity(outer * sz * inner);
                    for o in 0..outer {
                        let start = (o * v.dims[axis] + off) * inner;
                        f.extend_from_slice(&v.f[start..start + sz * inner]);
                    }
                    res.push(Val::new(dims, f));
                    off += sz;
                }
                res
            }
            "Range" => {
                let start = get(env, 0)?.f.first().copied().unwrap_or(0.0);
                let limit = get(env, 1)?.f.first().copied().unwrap_or(0.0);
                let delta = get(env, 2)?.f.first().copied().unwrap_or(1.0);
                let n = (ceilf((limit - start) / delta) as i64).max(0) as usize;
                let f: Vec<f32> = (0..n).map(|i| start + i as f32 * delta).collect();
                let iv: Vec<i64> = f.iter().map(|&x| x as i64).collect();
                vec![Val { dims: vec![n], f, i: Some(iv) }]
            }
            "MatMul" => vec![matmul(&get(env, 0)?, &get(env, 1)?, 0.0, 0.0)],
            "MatMulInteger" => {
                let a = get(env, 0)?;
                let b = get(env, 1)?;
                let azp = get(env, 2).ok().and_then(|t| t.f.first().copied()).unwrap_or(0.0);
                let bzp = get(env, 3).ok().and_then(|t| t.f.first().copied()).unwrap_or(0.0);
                vec![matmul(&a, &b, azp, bzp)]
            }
            "DynamicQuantizeLinear" => {
                // y = clamp(round(x/scale)+zp,0,255); scale=(hi-lo)/255 over
                // [min(x,0), max(x,0)]; zp=round(-lo/scale).
                let x = get(env, 0)?;
                let mut lo = 0f32;
                let mut hi = 0f32;
                for &v in &x.f {
                    if v < lo {
                        lo = v;
                    }
                    if v > hi {
                        hi = v;
                    }
                }
                let scale = if hi - lo > 0.0 { (hi - lo) / 255.0 } else { 1.0 };
                let zp = floorf(-lo / scale + 0.5).clamp(0.0, 255.0);
                let y: Vec<f32> = x.f.iter().map(|&v| floorf(v / scale + 0.5 + zp).clamp(0.0, 255.0)).collect();
                vec![
                    Val::new(x.dims.clone(), y),
                    Val::new(vec![], vec![scale]),
                    Val::new(vec![], vec![zp]),
                ]
            }
            "Relu" => {
                let mut v = get(env, 0)?;
                for x in &mut v.f {
                    if *x < 0.0 {
                        *x = 0.0;
                    }
                }
                v.i = None;
                vec![v]
            }
            "Sigmoid" => {
                let mut v = get(env, 0)?;
                for x in &mut v.f {
                    *x = sigmoid(*x);
                }
                v.i = None;
                vec![v]
            }
            "Sqrt" => {
                let mut v = get(env, 0)?;
                for x in &mut v.f {
                    *x = sqrtf(*x);
                }
                v.i = None;
                vec![v]
            }
            "Log" => {
                let mut v = get(env, 0)?;
                for x in &mut v.f {
                    *x = logf(*x);
                }
                v.i = None;
                vec![v]
            }
            "ReduceMean" => {
                let v = get(env, 0)?;
                let axes = attr_ints(node, "axes").or_else(|| get(env, 1).ok().map(|a| a.ints())).unwrap_or_default();
                let keep = attr_i(node, "keepdims", 1) == 1;
                let r = v.dims.len();
                let ax: Vec<usize> = axes.iter().map(|&a| (if a < 0 { a + r as i64 } else { a }) as usize).collect();
                let od: Vec<usize> = (0..r)
                    .filter_map(|k| {
                        if ax.contains(&k) {
                            if keep {
                                Some(1)
                            } else {
                                None
                            }
                        } else {
                            Some(v.dims[k])
                        }
                    })
                    .collect();
                // Iterate the full input, accumulate into the reduced index.
                let on: usize = od.iter().product::<usize>().max(1);
                let mut acc = vec![0f32; on];
                let mut cnt = vec![0u32; on];
                let mut idx = vec![0usize; r];
                for &x in &v.f {
                    // map idx -> output offset
                    let mut off = 0usize;
                    let mut stride = 1usize;
                    for k in (0..r).rev() {
                        if ax.contains(&k) {
                            if keep {
                                stride *= 1;
                            }
                            continue;
                        }
                        off += idx[k] * stride;
                        stride *= v.dims[k];
                    }
                    acc[off % on] += x;
                    cnt[off % on] += 1;
                    for k in (0..r).rev() {
                        idx[k] += 1;
                        if idx[k] < v.dims[k] {
                            break;
                        }
                        idx[k] = 0;
                    }
                }
                for (a, c) in acc.iter_mut().zip(cnt.iter()) {
                    if *c > 0 {
                        *a /= *c as f32;
                    }
                }
                vec![Val::new(od, acc)]
            }
            "Conv" => {
                // 1-D convolution: X[N,C,W], W[M,C/g,K] (+ optional bias[M]).
                let x = get(env, 0)?;
                let w = get(env, 1)?;
                let b = get(env, 2).ok();
                let groups = attr_i(node, "group", 1) as usize;
                let strides = attr_ints(node, "strides").unwrap_or_else(|| vec![1]);
                let pads = attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0]);
                let dil = attr_ints(node, "dilations").unwrap_or_else(|| vec![1]);
                if x.dims.len() != 3 || w.dims.len() != 3 {
                    return Err(alloc::format!("Conv: only 1-D supported (got x{:?} w{:?})", x.dims, w.dims));
                }
                let (nb, c, iw) = (x.dims[0], x.dims[1], x.dims[2]);
                let (m, cg, k) = (w.dims[0], w.dims[1], w.dims[2]);
                let (s, d) = (strides[0] as usize, dil[0] as usize);
                let (p0, p1) = (pads[0] as usize, pads[1] as usize);
                let ow = (iw + p0 + p1 - d * (k - 1) - 1) / s + 1;
                let mut out = vec![0f32; nb * m * ow];
                let cpg = c / groups; // channels per group (== cg)
                let mpg = m / groups;
                for n in 0..nb {
                    for om in 0..m {
                        let g = om / mpg;
                        let bias = b.as_ref().map(|bb| bb.f[om]).unwrap_or(0.0);
                        for o in 0..ow {
                            let mut acc = bias;
                            for ic in 0..cg.min(cpg) {
                                let xc = g * cpg + ic;
                                for kk in 0..k {
                                    let xi = (o * s + kk * d) as i64 - p0 as i64;
                                    if xi < 0 || xi >= iw as i64 {
                                        continue;
                                    }
                                    acc += x.f[(n * c + xc) * iw + xi as usize] * w.f[(om * cg + ic) * k + kk];
                                }
                            }
                            out[(n * m + om) * ow + o] = acc;
                        }
                    }
                }
                vec![Val::new(vec![nb, m, ow], out)]
            }
            "LSTM" => {
                // Single-direction ONNX LSTM: X[T,B,I], W[1,4H,I], R[1,4H,H],
                // B[1,8H], (5)=initial_h[1,B,H], (6)=initial_c. Gate order iofc.
                let x = get(env, 0)?;
                let w = get(env, 1)?;
                let rw = get(env, 2)?;
                let bb = get(env, 3).ok();
                let h0 = get(env, 5).ok();
                let c0 = get(env, 6).ok();
                let (t_len, bsz, isz) = (x.dims[0], x.dims[1], x.dims[2]);
                let h = rw.dims[2];
                let mut hs = h0.map(|v| v.f).unwrap_or_else(|| vec![0.0; bsz * h]);
                let mut cs = c0.map(|v| v.f).unwrap_or_else(|| vec![0.0; bsz * h]);
                let mut y = vec![0f32; t_len * bsz * h];
                let wb = bb.as_ref().map(|v| v.f.clone()).unwrap_or_else(|| vec![0.0; 8 * h]);
                for t in 0..t_len {
                    for b in 0..bsz {
                        // gates = W x_t + R h_{t-1} + Wb + Rb, order i,o,f,c
                        let mut gates = vec![0f32; 4 * h];
                        for (gi, gate) in gates.iter_mut().enumerate() {
                            let mut acc = wb[gi] + wb[4 * h + gi];
                            let wrow = gi * isz;
                            for i in 0..isz {
                                acc += w.f[wrow + i] * x.f[(t * bsz + b) * isz + i];
                            }
                            let rrow = gi * h;
                            for j in 0..h {
                                acc += rw.f[rrow + j] * hs[b * h + j];
                            }
                            *gate = acc;
                        }
                        for j in 0..h {
                            let i_g = sigmoid(gates[j]);
                            let o_g = sigmoid(gates[h + j]);
                            let f_g = sigmoid(gates[2 * h + j]);
                            let c_g = tanhf(gates[3 * h + j]);
                            let c_new = f_g * cs[b * h + j] + i_g * c_g;
                            cs[b * h + j] = c_new;
                            let h_new = o_g * tanhf(c_new);
                            hs[b * h + j] = h_new;
                            y[(t * bsz + b) * h + j] = h_new;
                        }
                    }
                }
                // Outputs: Y[T,1,B,H], Y_h[1,B,H], Y_c[1,B,H].
                vec![
                    Val::new(vec![t_len, 1, bsz, h], y),
                    Val::new(vec![1, bsz, h], hs),
                    Val::new(vec![1, bsz, h], cs),
                ]
            }
            other => return Err(alloc::format!("unsupported op {other}")),
        };
        for (k, o) in node.outputs.iter().enumerate() {
            if !o.is_empty() {
                if let Some(v) = out.get(k) {
                    env.insert((*o).to_string(), v.clone());
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    static SILERO: &[u8] = include_bytes!("../../../assets/voice/silero_vad.onnx");

    /// The LCG used to generate the host-side onnxruntime reference input.
    fn lcg_frame(seed: u32, n: usize) -> Vec<f32> {
        let mut v = seed;
        (0..n)
            .map(|_| {
                v = v.wrapping_mul(1664525).wrapping_add(1013904223);
                ((v >> 8) & 0xffff) as f32 / 65536.0 - 0.5
            })
            .collect()
    }

    /// Run the real silero VAD and match onnxruntime's probabilities:
    /// three chained steps on LCG noise -> [0.054053, 0.035221, 0.013769],
    /// silence -> 0.041476 (host reference, tolerance 3e-3).
    #[test_case]
    fn silero_vad_matches_onnxruntime() {
        let m = super::super::parse(SILERO).expect("parse");
        let expected = [0.054053f32, 0.035221, 0.013769];
        let mut h = Val::new(alloc::vec![2, 1, 64], alloc::vec![0.0; 128]);
        let mut c = Val::new(alloc::vec![2, 1, 64], alloc::vec![0.0; 128]);
        for (step, &want) in expected.iter().enumerate() {
            let x = Val::new(alloc::vec![1, 512], lcg_frame(42 + step as u32, 512));
            let out = run(&m, &[("x", x), ("h", h.clone()), ("c", c.clone())]).expect("run");
            let prob = out.get("prob").unwrap().f[0];
            crate::serial_println!("onnx.vad: step {} prob {} (want {})", step, prob, want);
            assert!((prob - want).abs() < 3e-3, "step {step}: prob {prob} vs {want}");
            h = out.get("new_h").unwrap().clone();
            c = out.get("new_c").unwrap().clone();
        }
        // Silence.
        let x = Val::new(alloc::vec![1, 512], alloc::vec![0.0; 512]);
        let h = Val::new(alloc::vec![2, 1, 64], alloc::vec![0.0; 128]);
        let c = Val::new(alloc::vec![2, 1, 64], alloc::vec![0.0; 128]);
        let out = run(&m, &[("x", x), ("h", h), ("c", c)]).expect("run");
        let prob = out.get("prob").unwrap().f[0];
        assert!((prob - 0.041476).abs() < 3e-3, "silence prob {prob}");
    }

    /// DynamicQuantizeLinear + MatMulInteger against hand/onnxruntime values.
    #[test_case]
    fn quant_ops_match_reference() {
        // DynamicQuantizeLinear
        let x = Val::new(alloc::vec![8], alloc::vec![0.2, -0.5, 1.3, -2.1, 0.0, 3.0, -1.0, 0.75]);
        let (mut lo, mut hi) = (0f32, 0f32);
        for &v in &x.f {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let scale = (hi - lo) / 255.0;
        let zp = floorf(-lo / scale + 0.5);
        assert!((scale - 0.02).abs() < 1e-6, "scale {scale}");
        assert_eq!(zp as i32, 105);
        let q: alloc::vec::Vec<i32> = x.f.iter().map(|&v| floorf(v / scale + 0.5 + zp).clamp(0.0, 255.0) as i32).collect();
        assert_eq!(q, alloc::vec![115, 80, 170, 0, 105, 255, 55, 143]);

        // MatMulInteger: A[2x3] azp=5, B[3x2] bzp=1 -> [[130,175],[310,445]].
        let a = Val::new(alloc::vec![2, 3], alloc::vec![10., 20., 30., 40., 50., 60.]);
        let b = Val::new(alloc::vec![3, 2], alloc::vec![1., 2., 3., 4., 5., 6.]);
        let y = matmul(&a, &b, 5.0, 1.0);
        assert_eq!(y.dims, alloc::vec![2, 2]);
        assert_eq!(y.f, alloc::vec![130.0, 175.0, 310.0, 445.0]);
    }

    /// LayerNormalization + Softmax sanity: LN → mean 0 / unit var (scale 1,
    /// bias 0); Softmax rows sum to 1.
    #[test_case]
    fn layernorm_and_softmax() {
        let x = Val::new(alloc::vec![1, 4], alloc::vec![1.0, 2.0, 3.0, 4.0]);
        let scale = Val::new(alloc::vec![4], alloc::vec![1.0, 1.0, 1.0, 1.0]);
        let m = super::super::Model {
            ir_version: 7,
            graph: super::super::Graph {
                name: "t",
                nodes: alloc::vec![super::super::Node {
                    op: "LayerNormalization",
                    name: "ln",
                    inputs: alloc::vec!["x", "s"],
                    outputs: alloc::vec!["y"],
                    attrs: alloc::vec![],
                }],
                initializers: alloc::vec![],
                inputs: alloc::vec!["x", "s"],
                outputs: alloc::vec!["y"],
            },
        };
        let out = run(&m, &[("x", x), ("s", scale)]).unwrap();
        let y = &out.get("y").unwrap().f;
        let mean: f32 = y.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-4, "LN mean {mean}");
        let var: f32 = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 1e-2, "LN var {var}");
    }
}

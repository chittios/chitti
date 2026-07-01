#!/usr/bin/env python3
"""Host-first reconstruction of the Qwen3.5-0.8B hybrid (Gated-DeltaNet +
gated full-attention) forward pass in NumPy. Purpose: determine whether the
architecture can be reproduced offline (coherent greedy output) BEFORE
attempting the bare-metal kernel port. If this can't produce coherent text,
the kernel port is not worth attempting.

This is a best-effort reconstruction from the GGUF tensor structure; several
constants (gated-delta recurrence, gate activations) are inferred.
"""
import struct
import numpy as np

GGML_F32, GGML_Q8_0 = 0, 8
QK = 32


class Gguf:
    def __init__(self, path):
        self.buf = open(path, "rb").read()
        self.pos = 0
        assert self._u32() == 0x46554747
        assert self._u32() == 3
        nt = self._u64()
        nkv = self._u64()
        self.meta = {}
        self.tokens = []
        for _ in range(nkv):
            k = self._str()
            t = self._u32()
            v = self._value(t)
            self.meta[k] = v
            if k == "tokenizer.ggml.tokens":
                self.tokens = v
        self.tensors = {}
        for _ in range(nt):
            name = self._str()
            nd = self._u32()
            dims = [self._u64() for _ in range(nd)]
            tt = self._u32()
            off = self._u64()
            self.tensors[name] = (dims, tt, off)
        align = self.meta.get("general.alignment", 32)
        self.data_base = (self.pos + align - 1) & ~(align - 1)

    def _take(self, n):
        s = self.buf[self.pos:self.pos + n]
        self.pos += n
        return s

    def _u32(self):
        return struct.unpack("<I", self._take(4))[0]

    def _u64(self):
        return struct.unpack("<Q", self._take(8))[0]

    def _str(self):
        return self._take(self._u64()).decode("utf-8", "replace")

    def _value(self, t):
        if t == 8:
            return self._str()
        if t == 6:
            return struct.unpack("<f", self._take(4))[0]
        if t in (4, 5):
            return struct.unpack("<i", self._take(4))[0]
        if t in (10, 11):
            return struct.unpack("<q", self._take(8))[0]
        if t in (7, 0):
            return self._take(1)[0]
        if t == 1:
            return struct.unpack("<b", self._take(1))[0]
        if t in (2, 3):
            return struct.unpack("<h", self._take(2))[0]
        if t == 12:
            return struct.unpack("<d", self._take(8))[0]
        if t == 9:
            et = self._u32()
            n = self._u64()
            return [self._value(et) for _ in range(n)]
        raise ValueError(t)

    def t(self, name):
        dims, tt, off = self.tensors[name]
        base = self.data_base + off
        n = 1
        for d in dims:
            n *= d
        if tt == GGML_F32:
            arr = np.frombuffer(self.buf, np.float32, n, base).astype(np.float32)
        elif tt == GGML_Q8_0:
            nb = n // QK
            raw = np.frombuffer(self.buf, np.uint8, nb * 34, base).reshape(nb, 34)
            d = raw[:, :2].copy().view(np.float16).astype(np.float32)
            q = raw[:, 2:].view(np.int8).astype(np.float32)
            arr = (d * q).reshape(n)
        else:
            raise ValueError(f"{name}: type {tt}")
        if len(dims) == 2:
            return arr.reshape(dims[1], dims[0])
        return arr


def rms(x, w, eps=1e-6):
    x = x.astype(np.float32)
    return x / np.sqrt(np.mean(x * x) + eps) * w


def rms_head(x, w, eps=1e-6):
    # x: (heads, dim), w: (dim,)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="assets/model.gguf")
    ap.add_argument("--prompt", default="Hello, my name is")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    g = Gguf(args.model)
    m = g.meta
    P = "qwen35."
    dim = m[P + "embedding_length"]
    n_layers = m[P + "block_count"]
    ffn = m[P + "feed_forward_length"]
    eps = np.float32(m[P + "attention.layer_norm_rms_epsilon"])
    theta = np.float32(m[P + "rope.freq_base"])
    rope_dim = m[P + "rope.dimension_count"]  # 64
    attn_interval = m[P + "full_attention_interval"]  # 4
    head_dim = m[P + "attention.key_length"]  # 256
    n_kv = m[P + "attention.head_count_kv"]  # 2
    ssm_heads = m[P + "ssm.group_count"]  # 16
    ssm_inner = m[P + "ssm.inner_size"]  # 2048
    ssm_head_dim = ssm_inner // ssm_heads  # 128
    print(f"dim={dim} layers={n_layers} ffn={ffn} head_dim={head_dim} n_kv={n_kv} "
          f"ssm_heads={ssm_heads} ssm_head_dim={ssm_head_dim} rope_dim={rope_dim} attn_every={attn_interval}")

    tok_embd = g.t("token_embd.weight")
    out_norm = g.t("output_norm.weight")

    def is_attn(l):
        return (l % attn_interval) == (attn_interval - 1)

    def partial_rope(vec, pos):
        # rotate first rope_dim dims (rope_dim/2 pairs), NeoX pairing.
        half = rope_dim // 2
        out = vec.copy()
        for i in range(half):
            freq = 1.0 / (float(theta) ** (float(2 * i) / rope_dim))
            ang = pos * freq
            c, s = np.cos(ang), np.sin(ang)
            a, b = vec[i], vec[i + half]
            out[i] = a * c - b * s
            out[i + half] = b * c + a * s
        return out

    # BPE tokenizer (reuse from ref_forward via inline minimal version)
    import re, ref_forward
    ids = ref_forward.tokenize(g, args.prompt)
    print(f"prompt ids: {ids}")

    n_q_heads = 8  # from head_count; q_proj = query(2048=8*256) + gate(2048)

    # Per-layer weights loader (lazy dict).
    def W(l, name):
        return g.t(f"blk.{l}.{name}")

    seqlen = len(ids) + args.n
    # deltanet recurrent state per layer: (ssm_heads, ssm_head_dim, ssm_head_dim)
    delta_state = {l: np.zeros((ssm_heads, ssm_head_dim, ssm_head_dim), np.float32)
                   for l in range(n_layers) if not is_attn(l)}
    conv_state = {l: np.zeros((4, 6144), np.float32) for l in range(n_layers) if not is_attn(l)}
    kv_cache = {l: ([], []) for l in range(n_layers) if is_attn(l)}

    def step(token, pos):
        x = tok_embd[token].astype(np.float32).copy()
        for l in range(n_layers):
            xn = rms(x, W(l, "attn_norm.weight"), eps)
            if is_attn(l):
                # q_proj outputs query+gate INTERLEAVED per head: [q256,gate256]x8
                qg = (W(l, "attn_q.weight") @ xn).reshape(n_q_heads, 2 * head_dim)
                q = qg[:, :head_dim]
                gate = qg[:, head_dim:]
                k = (W(l, "attn_k.weight") @ xn).reshape(n_kv, head_dim)
                v = (W(l, "attn_v.weight") @ xn).reshape(n_kv, head_dim)
                q = rms_head(q, W(l, "attn_q_norm.weight"), eps)
                k = rms_head(k, W(l, "attn_k_norm.weight"), eps)
                q = np.stack([partial_rope(q[h], pos) for h in range(n_q_heads)])
                k = np.stack([partial_rope(k[h], pos) for h in range(n_kv)])
                kc, vc = kv_cache[l]
                kc.append(k)
                vc.append(v)
                grp = n_q_heads // n_kv
                o = np.zeros((n_q_heads, head_dim), np.float32)
                scale = 1.0 / np.sqrt(head_dim)
                for h in range(n_q_heads):
                    kvh = h // grp
                    scores = np.array([np.dot(q[h], kc[t][kvh]) * scale for t in range(pos + 1)], np.float32)
                    scores = scores - scores.max()
                    w = np.exp(scores)
                    w /= w.sum()
                    o[h] = sum(w[t] * vc[t][kvh] for t in range(pos + 1))
                o = o * (1.0 / (1.0 + np.exp(-gate)))  # sigmoid output gate
                out = W(l, "attn_output.weight") @ o.reshape(-1)
                x = x + out
            else:
                qkv = W(l, "attn_qkv.weight") @ xn  # 6144
                # causal depthwise conv (kernel 4) + silu
                cs = conv_state[l]
                cs[:-1] = cs[1:]
                cs[-1] = qkv
                convw = W(l, "ssm_conv1d.weight")  # stored (6144,4) after reshape
                conv = np.sum(cs * convw.T, axis=0)
                conv = silu(conv)
                q, k, vv = conv[:2048].reshape(ssm_heads, ssm_head_dim), \
                    conv[2048:4096].reshape(ssm_heads, ssm_head_dim), \
                    conv[4096:].reshape(ssm_heads, ssm_head_dim)
                # L2 normalize q,k per head
                # L2-norm q,k (rsqrt(sum sq + eps)), then scale q by 1/sqrt(head_k_dim).
                q = q / np.sqrt(np.sum(q * q, -1, keepdims=True) + 1e-6)
                k = k / np.sqrt(np.sum(k * k, -1, keepdims=True) + 1e-6)
                q = q * (1.0 / np.sqrt(ssm_head_dim))
                beta = 1.0 / (1.0 + np.exp(-(W(l, "ssm_beta.weight") @ xn)))  # (16)
                dt = softplus(W(l, "ssm_alpha.weight") @ xn + W(l, "ssm_dt.bias"))  # (16)
                a = -np.exp(W(l, "ssm_a"))  # (16)
                g_decay = np.exp(a * dt)  # (16) in (0,1)
                S = delta_state[l]
                o = np.zeros((ssm_heads, ssm_head_dim), np.float32)
                for h in range(ssm_heads):
                    Sh = g_decay[h] * S[h]
                    kh = k[h]
                    # delta rule: S += beta * outer(k, v - S^T k)
                    err = vv[h] - Sh.T @ kh
                    Sh = Sh + beta[h] * np.outer(kh, err)
                    S[h] = Sh
                    o[h] = Sh.T @ q[h]
                # Gated RMSNorm: normalize per head, multiply by SiLU(z), z=attn_gate@x.
                z = (W(l, "attn_gate.weight") @ xn).reshape(ssm_heads, ssm_head_dim)
                o = rms_head(o, W(l, "ssm_norm.weight"), eps) * silu(z)
                out = W(l, "ssm_out.weight") @ o.reshape(-1)
                x = x + out
            xn2 = rms(x, W(l, "post_attention_norm.weight"), eps)
            gg = W(l, "ffn_gate.weight") @ xn2
            up = W(l, "ffn_up.weight") @ xn2
            x = x + W(l, "ffn_down.weight") @ (silu(gg) * up)
        xn = rms(x, out_norm, eps)
        return tok_embd @ xn  # tied output

    logits = None
    for pos, tok in enumerate(ids):
        logits = step(tok, pos)
    logits_final = logits.copy()  # prompt-final logits (before generation)
    gen = []
    pos = len(ids)
    for _ in range(args.n):
        nxt = int(np.argmax(logits))
        gen.append(nxt)
        logits = step(nxt, pos)
        pos += 1

    def detok(idl):
        b2u = ref_forward.bytes_to_unicode()
        u2b = {v: k for k, v in b2u.items()}
        out = bytearray()
        for i in idl:
            for ch in g.tokens[i]:
                out.append(u2b.get(ch, ord(ch) if ord(ch) < 256 else 63))
        return bytes(out).decode("utf-8", "replace")

    print(f"gen ids: {gen}")
    print(f"continuation: {detok(gen)!r}")
    print(f"full: {(args.prompt + detok(gen))!r}")

    # Emit refcheck consts for the kernel port to reproduce.
    import json
    ids_lit = ", ".join(str(i) for i in ids)
    gen_lit = ", ".join(str(i) for i in gen)
    rs = f"""//! GENERATED by tools/ref_qwen35.py -- do not edit by hand.
//! Fixed prompt + greedy reference continuation for the Qwen3.5-0.8B
//! Phase 3 parity gate (`cargo xtask ref-check`) and the boot-time demo.
pub const PROMPT: &str = {json.dumps(args.prompt)};
pub static PROMPT_IDS: [u32; {len(ids)}] = [{ids_lit}];
pub static EXPECTED_CONTINUATION: [u32; {len(gen)}] = [{gen_lit}];
pub const PROMPT_FINAL_ARGMAX: u32 = {gen[0]};
pub const PROMPT_FINAL_LOGIT: f32 = {float(logits_final[gen[0]])!r};
"""
    with open("kernel/src/cortex/refcheck.rs", "w") as f:
        f.write(rs)
    print("wrote kernel/src/cortex/refcheck.rs")


if __name__ == "__main__":
    main()

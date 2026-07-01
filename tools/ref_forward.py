#!/usr/bin/env python3
"""Phase 3 reference forward pass (CHITTI_OS_HANDOFF.md Part 2 parity gate).

Loads the SAME Qwen2.5-0.5B GGUF the kernel boots, runs a float32
decoder-only forward pass with numerics matching cortex/model.rs
(dequant Q8_0 -> f32, RMSNorm, NeoX RoPE theta=1e6, GQA, SwiGLU), greedy-
decodes N tokens from a fixed prompt, and emits:

  * kernel/src/cortex/refcheck.rs -- the prompt token ids + expected greedy
    continuation ids + a couple of logit anchors, as Rust consts, for the
    in-kernel demo and `cargo xtask ref-check` to reproduce and compare.
  * stdout -- a human-readable summary (detokenized prompt + continuation).

Run: python3 tools/ref_forward.py [--prompt "..."] [--n 8]
"""
import argparse
import json
import struct
import numpy as np

GGML_F32, GGML_F16, GGML_Q8_0 = 0, 1, 8
QK = 32


# --- GGUF reader (mirrors cortex/gguf.rs) ---------------------------------
class Gguf:
    def __init__(self, path):
        self.buf = open(path, "rb").read()
        self.pos = 0
        magic = self._u32()
        assert magic == 0x46554747, "bad magic"
        ver = self._u32()
        assert ver == 3, ver
        n_tensors = self._u64()
        n_kv = self._u64()
        self.meta = {}
        self.tokens = []
        for _ in range(n_kv):
            key = self._str()
            vt = self._u32()
            val = self._value(vt)
            self.meta[key] = val
            if key == "tokenizer.ggml.tokens":
                self.tokens = val
        self.tensors = {}
        for _ in range(n_tensors):
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

    def _i32(self):
        return struct.unpack("<i", self._take(4))[0]

    def _u64(self):
        return struct.unpack("<Q", self._take(8))[0]

    def _f32(self):
        return struct.unpack("<f", self._take(4))[0]

    def _str(self):
        n = self._u64()
        return self._take(n).decode("utf-8", "replace")

    def _value(self, vt):
        if vt == 8:
            return self._str()
        if vt == 6:
            return self._f32()
        if vt == 4:
            return self._u32()
        if vt == 5:
            return self._i32()
        if vt == 10:
            return self._u64()
        if vt == 11:
            return struct.unpack("<q", self._take(8))[0]
        if vt == 7:
            return self._take(1)[0]
        if vt == 0:
            return self._take(1)[0]
        if vt == 1:
            return struct.unpack("<b", self._take(1))[0]
        if vt == 2:
            return struct.unpack("<H", self._take(2))[0]
        if vt == 3:
            return struct.unpack("<h", self._take(2))[0]
        if vt == 12:
            return struct.unpack("<d", self._take(8))[0]
        if vt == 9:
            et = self._u32()
            n = self._u64()
            return [self._value(et) for _ in range(n)]
        raise ValueError(vt)

    def tensor_f32(self, name):
        dims, tt, off = self.tensors[name]
        base = self.data_base + off
        n = 1
        for d in dims:
            n *= d
        if tt == GGML_F32:
            arr = np.frombuffer(self.buf, dtype=np.float32, count=n, offset=base).astype(np.float32)
        elif tt == GGML_Q8_0:
            nblocks = n // QK
            raw = np.frombuffer(self.buf, dtype=np.uint8, count=nblocks * 34, offset=base)
            raw = raw.reshape(nblocks, 34)
            d = raw[:, :2].copy().view(np.float16).astype(np.float32)  # (nblocks,1)
            q = raw[:, 2:].view(np.int8).astype(np.float32)  # (nblocks,32)
            arr = (d * q).reshape(n)
        else:
            raise ValueError(f"{name}: unsupported type {tt}")
        # dims are [fastest, ...]; a matrix [n_cols, n_rows] -> (n_rows, n_cols)
        if len(dims) == 2:
            return arr.reshape(dims[1], dims[0])
        return arr


# --- byte-level BPE tokenizer (GPT-2 style, from tokens + merges) ----------
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


import re

PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""".replace(
        r"\p{L}", r"[^\W\d_]"
    ).replace(r"\p{N}", r"\d"),
    re.UNICODE,
)


def get_pairs(word):
    return set(zip(word[:-1], word[1:]))


def tokenize(gguf, text):
    b2u = bytes_to_unicode()
    vocab = {t: i for i, t in enumerate(gguf.tokens)}
    merges = gguf.meta.get("tokenizer.ggml.merges", [])
    ranks = {tuple(m.split(" ")): i for i, m in enumerate(merges)}
    ids = []
    for piece in PAT.findall(text):
        token = "".join(b2u[b] for b in piece.encode("utf-8"))
        word = list(token)
        while len(word) > 1:
            pairs = get_pairs(word)
            best = min(pairs, key=lambda p: ranks.get(p, 1e18))
            if best not in ranks:
                break
            first, second = best
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        for sym in word:
            if sym in vocab:
                ids.append(vocab[sym])
            else:
                # fall back to byte tokens (should be rare for ASCII)
                for ch in sym:
                    ids.append(vocab.get(ch, 0))
    return ids


# --- forward pass (mirrors cortex/model.rs) --------------------------------
def rmsnorm(x, w, eps):
    x = x.astype(np.float32)
    ss = np.float32(np.mean(x * x))
    return (x * np.float32(1.0 / np.sqrt(ss + eps)) * w).astype(np.float32)


def rope(vec, pos, head_dim, theta):
    half = head_dim // 2
    out = vec.copy().astype(np.float32)
    for i in range(half):
        freq = np.float32(1.0 / (float(theta) ** (float(2 * i) / head_dim)))
        angle = np.float32(pos * freq)
        c, s = np.float32(np.cos(angle)), np.float32(np.sin(angle))
        a, b = np.float32(vec[i]), np.float32(vec[i + half])
        out[i] = a * c - b * s
        out[i + half] = b * c + a * s
    return out


def softmax(x):
    x = x.astype(np.float32)
    m = np.max(x)
    e = np.exp((x - m).astype(np.float32)).astype(np.float32)
    return (e / np.sum(e, dtype=np.float32)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="assets/model.gguf")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    g = Gguf(args.model)
    m = g.meta
    dim = m["qwen2.embedding_length"]
    n_layers = m["qwen2.block_count"]
    n_heads = m["qwen2.attention.head_count"]
    n_kv = m["qwen2.attention.head_count_kv"]
    head_dim = dim // n_heads
    kv_dim = n_kv * head_dim
    ffn = m["qwen2.feed_forward_length"]
    eps = np.float32(m["qwen2.attention.layer_norm_rms_epsilon"])
    theta = np.float32(m.get("qwen2.rope.freq_base", 1e6))
    group = n_heads // n_kv

    print(f"config: dim={dim} layers={n_layers} heads={n_heads}/{n_kv} head_dim={head_dim} ffn={ffn}")

    tok_embd = g.tensor_f32("token_embd.weight")  # (vocab, dim)
    out_norm = g.tensor_f32("output_norm.weight")
    try:
        out_w = g.tensor_f32("output.weight")
    except KeyError:
        out_w = tok_embd
    layers = []
    for l in range(n_layers):
        p = f"blk.{l}."
        layers.append({
            "an": g.tensor_f32(p + "attn_norm.weight"),
            "qw": g.tensor_f32(p + "attn_q.weight"), "qb": g.tensor_f32(p + "attn_q.bias"),
            "kw": g.tensor_f32(p + "attn_k.weight"), "kb": g.tensor_f32(p + "attn_k.bias"),
            "vw": g.tensor_f32(p + "attn_v.weight"), "vb": g.tensor_f32(p + "attn_v.bias"),
            "ow": g.tensor_f32(p + "attn_output.weight"),
            "fn": g.tensor_f32(p + "ffn_norm.weight"),
            "gw": g.tensor_f32(p + "ffn_gate.weight"),
            "uw": g.tensor_f32(p + "ffn_up.weight"),
            "dw": g.tensor_f32(p + "ffn_down.weight"),
        })

    prompt_ids = tokenize(g, args.prompt)
    print(f"prompt {args.prompt!r} -> ids {prompt_ids}")

    # KV cache: per layer lists of k,v vectors.
    kcache = [[] for _ in range(n_layers)]
    vcache = [[] for _ in range(n_layers)]

    def step(token, pos):
        h = tok_embd[token].astype(np.float32).copy()
        for l in range(n_layers):
            L = layers[l]
            xn = rmsnorm(h, L["an"], eps)
            q = (L["qw"] @ xn + L["qb"]).astype(np.float32)
            k = (L["kw"] @ xn + L["kb"]).astype(np.float32)
            v = (L["vw"] @ xn + L["vb"]).astype(np.float32)
            for hh in range(n_heads):
                q[hh * head_dim:(hh + 1) * head_dim] = rope(q[hh * head_dim:(hh + 1) * head_dim], pos, head_dim, theta)
            for hh in range(n_kv):
                k[hh * head_dim:(hh + 1) * head_dim] = rope(k[hh * head_dim:(hh + 1) * head_dim], pos, head_dim, theta)
            kcache[l].append(k)
            vcache[l].append(v)
            attn = np.zeros(dim, dtype=np.float32)
            scale = np.float32(1.0 / np.sqrt(head_dim))
            for hh in range(n_heads):
                kvh = hh // group
                qh = q[hh * head_dim:(hh + 1) * head_dim]
                scores = np.array([
                    np.float32(np.dot(qh, kcache[l][t][kvh * head_dim:(kvh + 1) * head_dim])) * scale
                    for t in range(pos + 1)
                ], dtype=np.float32)
                scores = softmax(scores)
                acc = np.zeros(head_dim, dtype=np.float32)
                for t in range(pos + 1):
                    acc += scores[t] * vcache[l][t][kvh * head_dim:(kvh + 1) * head_dim]
                attn[hh * head_dim:(hh + 1) * head_dim] = acc
            h = (h + (L["ow"] @ attn)).astype(np.float32)
            xn2 = rmsnorm(h, L["fn"], eps)
            gate = (L["gw"] @ xn2).astype(np.float32)
            up = (L["uw"] @ xn2).astype(np.float32)
            act = (gate / (1.0 + np.exp(-gate.astype(np.float32))) * up).astype(np.float32)
            h = (h + (L["dw"] @ act)).astype(np.float32)
        xn = rmsnorm(h, out_norm, eps)
        return (out_w @ xn).astype(np.float32)

    logits = None
    for pos, tok in enumerate(prompt_ids):
        logits = step(tok, pos)
    gen = []
    prompt_last_argmax = int(np.argmax(logits))
    prompt_last_max = float(logits[prompt_last_argmax])
    pos = len(prompt_ids)
    for _ in range(args.n):
        nxt = int(np.argmax(logits))
        gen.append(nxt)
        logits = step(nxt, pos)
        pos += 1

    def detok(ids):
        b2u = bytes_to_unicode()
        u2b = {v: k for k, v in b2u.items()}
        out = bytearray()
        for i in ids:
            for ch in g.tokens[i]:
                out.append(u2b.get(ch, ord(ch) if ord(ch) < 256 else 63))
        return bytes(out).decode("utf-8", "replace")

    print(f"greedy continuation ids: {gen}")
    print(f"continuation text: {detok(gen)!r}")
    print(f"full: {(args.prompt + detok(gen))!r}")
    print(f"prompt-final argmax id={prompt_last_argmax} logit={prompt_last_max:.6f} tok={detok([prompt_last_argmax])!r}")

    # Emit the Rust refcheck consts.
    ids_lit = ", ".join(str(i) for i in prompt_ids)
    gen_lit = ", ".join(str(i) for i in gen)
    rs = f"""//! GENERATED by tools/ref_forward.py -- do not edit by hand.
//! Fixed prompt + greedy reference continuation for the Phase 3 parity
//! gate (`cargo xtask ref-check`) and the boot-time inference demo.
pub const PROMPT: &str = {json.dumps(args.prompt)};
pub static PROMPT_IDS: [u32; {len(prompt_ids)}] = [{ids_lit}];
pub static EXPECTED_CONTINUATION: [u32; {len(gen)}] = [{gen_lit}];
pub const PROMPT_FINAL_ARGMAX: u32 = {prompt_last_argmax};
pub const PROMPT_FINAL_LOGIT: f32 = {prompt_last_max!r};
"""
    with open("kernel/src/cortex/refcheck.rs", "w") as f:
        f.write(rs)
    with open("target/refcheck_expected.json", "w") as f:
        json.dump({
            "prompt_ids": prompt_ids, "continuation": gen,
            "prompt_final_argmax": prompt_last_argmax, "prompt_final_logit": prompt_last_max,
        }, f)
    print("wrote kernel/src/cortex/refcheck.rs and target/refcheck_expected.json")


if __name__ == "__main__":
    main()

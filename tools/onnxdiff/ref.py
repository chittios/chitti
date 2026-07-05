#!/usr/bin/env python3
"""onnxruntime reference dump for onnxdiff: run a model with the same fixed
inputs as the Rust harness and print per-tensor stats for every intermediate.

  python3 ref.py kitten   <kitten.onnx>   > /tmp/ref_dump.txt
  python3 ref.py parakeet <parakeet.onnx> <mel.bin> > /tmp/ref_dump.txt
"""
import sys, struct
import numpy as np
import onnx
import onnxruntime as ort

HELLO_IDS = [0, 50, 83, 54, 156, 57, 135, 16, 65, 156, 87, 158, 54, 46, 0]

def voice_row(path, ntok):
    b = open(path, "rb").read()
    row = min(ntok, 399)
    return np.frombuffer(b[row*256*4:(row+1)*256*4], dtype="<f4").reshape(1, 256).copy()

def main():
    mode, model_path = sys.argv[1], sys.argv[2]
    m = onnx.load(model_path)
    # Expose every node output as a graph output so ORT reports intermediates.
    existing = {o.name for o in m.graph.output}
    for node in m.graph.node:
        for o in node.output:
            if o and o not in existing:
                m.graph.output.append(onnx.helper.make_empty_tensor_value_info(o))
                existing.add(o)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # keep node names
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])

    if mode == "kitten":
        import os
        vb = os.path.join(os.path.dirname(__file__), "../../kernel/src/sound/testdata/kitten_voice.bin")
        n = len(HELLO_IDS)
        feeds = {
            "input_ids": np.array([HELLO_IDS], dtype=np.int64),
            "style": voice_row(vb, n),
            "speed": np.array([1.0], dtype=np.float32),
        }
    elif mode == "parakeet":
        mel = np.frombuffer(open(sys.argv[3], "rb").read(), dtype="<f4")
        frames = len(mel) // 80
        feeds = {
            "audio_signal": mel.reshape(1, 80, frames).copy(),
            "length": np.array([frames], dtype=np.int64),
        }
    else:
        sys.exit("mode must be kitten|parakeet")

    names = [o.name for o in sess.get_outputs()]
    outs = sess.run(names, feeds)
    for name, v in zip(names, outs):
        try:
            a = np.asarray(v)
        except ValueError:
            print(f"REF '{name}' n=0 (sequence)")
            continue
        if a.dtype == object or a.size == 0:
            print(f"REF '{name}' n=0")
            continue
        f = a.astype(np.float64).ravel()
        head = ", ".join(f"{x:.6g}" for x in f[:4])
        print(f"REF '{name}' dims={list(a.shape)} n={a.size} maxabs={np.abs(f).max():.6f} mean={f.mean():.6f} v=[{head}]")

if __name__ == "__main__":
    main()

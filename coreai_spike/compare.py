"""Core AI vs the production backend for one model, same inputs, scored
against the ONNX Runtime CPU reference.

Run with the main repo's venv (it needs roop.inference for the production
path):
    venv/bin/python coreai_spike/compare.py <model.onnx> <model.aimodel> [--runs N]

Inputs: 4-D image inputs get smooth random content in the range the app
feeds that model; other inputs get a unit-norm random vector (the swappers'
source embedding).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import onnxruntime as ort

MAIN_REPO = "/Users/j0hnnyxm4s/Documents/GitHub/roop-mlx"
sys.path.insert(0, MAIN_REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "out", "coreai-bench")


def make_inputs(sess, lo, hi, seed=7):
    rng = np.random.default_rng(seed)
    feed = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        if len(shape) == 4:
            n, c, h, w = shape
            img = rng.random((h // 8 + 1, w // 8 + 1, c)).astype(np.float32)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            img = cv2.GaussianBlur(img, (0, 0), 1.5).reshape(h, w, c)
            img = np.clip(img, 0, 1) * (hi - lo) + lo
            feed[i.name] = img.transpose(2, 0, 1)[None].astype(np.float32)
        else:
            v = rng.normal(size=shape).astype(np.float32)
            feed[i.name] = v / np.linalg.norm(v)
    return feed


def score(out, ref):
    d = out.astype(np.float64) - ref.astype(np.float64)
    rng = float(ref.max() - ref.min()) or 1.0
    mse = float((d ** 2).mean())
    psnr = 10 * np.log10(rng * rng / mse) if mse > 0 else float("inf")
    return psnr, float(np.abs(d).max())


def time_session(sess, feed, runs):
    for _ in range(3):
        sess.run(None, feed)
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        out = sess.run(None, feed)
        t.append((time.perf_counter() - s) * 1000)
    t.sort()
    return out, t[len(t) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("aimodel")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--range", default="-1,1", help="image input range lo,hi")
    ap.add_argument("--units", default="default,ane,gpu,cpu")
    a = ap.parse_args()
    lo, hi = (float(v) for v in a.range.split(","))

    ort.set_default_logger_severity(3)
    ref_sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    feed = make_inputs(ref_sess, lo, hi)
    out_names = [o.name for o in ref_sess.get_outputs()]
    ref = dict(zip(out_names, ref_sess.run(None, feed)))

    rows = []
    from roop.inference import create_session
    prod = create_session(os.path.abspath(a.onnx))
    backend = type(prod).__name__
    if hasattr(prod, "get_providers"):
        backend = prod.get_providers()[0]
    outs, med = time_session(prod, feed, a.runs)
    rows.append((f"production ({backend})", med, None, None, dict(zip(out_names, outs))))

    with tempfile.TemporaryDirectory() as io:
        specs = []
        for name, arr in feed.items():
            fname = f"in_{len(specs)}.bin"
            np.ascontiguousarray(arr, dtype=np.float32).tofile(os.path.join(io, fname))
            specs.append({"name": name, "shape": list(arr.shape), "file": fname})
        with open(os.path.join(io, "manifest.json"), "w") as f:
            json.dump({"inputs": specs}, f)
        for unit in a.units.split(","):
            p = subprocess.run([BENCH, os.path.abspath(a.aimodel), io, unit, str(a.runs)],
                               capture_output=True, text=True)
            if p.returncode != 0:
                rows.append((f"Core AI {unit}", None, None, None, {"error": (p.stderr or p.stdout)[-300:]}))
                continue
            res = json.load(open(os.path.join(io, "result.json")))
            outs = {}
            for name in out_names:
                path = os.path.join(io, f"out_{name}.bin")
                if os.path.exists(path):
                    outs[name] = np.fromfile(path, dtype=np.float32).reshape(res["output_shapes"][name])
            rows.append((f"Core AI {unit}", res["median_ms"], res["load_ms"], res["first_run_ms"], outs))

    print(f"\n{os.path.basename(a.onnx)}  ({a.runs} timed runs each)")
    print(f"{'backend':34}{'median ms':>10}{'load ms':>10}{'1st run':>9}   accuracy vs ORT CPU per output")
    for label, med, load, first, outs in rows:
        if "error" in outs:
            print(f"{label:34}{'FAILED':>10}   {outs['error'].strip().splitlines()[-1][:110]}")
            continue
        acc = "  ".join(f"{n}: {score(outs[n], ref[n])[0]:.1f} dB (max {score(outs[n], ref[n])[1]:.3g})"
                        for n in out_names if n in outs)
        f = lambda v: f"{v:10.1f}" if v is not None else f"{'-':>10}"  # noqa: E731
        print(f"{label:34}{f(med)}{f(load)}{(f'{first:9.1f}' if first else f'{chr(45):>9}')}   {acc}")


if __name__ == "__main__":
    main()

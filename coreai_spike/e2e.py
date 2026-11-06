"""End-to-end from Python: production session vs CoreAISession, alternating
runs, all copies included; plus a 4-thread concurrency check.

    venv/bin/python coreai_spike/e2e.py   (main repo venv)
"""
import os
import sys
import threading
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/j0hnnyxm4s/Documents/GitHub/roop-mlx")
from compare import make_inputs, score  # noqa: E402
from coreai_session import CoreAISession  # noqa: E402
from roop.inference import create_session  # noqa: E402

M = "/Users/j0hnnyxm4s/Documents/GitHub/roop-mlx/models"
MODELS = [("hyperswap_1a_256", (-1, 1)), ("GFPGANv1.4", (-1, 1)),
          ("GPEN-BFR-512", (-1, 1)), ("inswapper_128", (0, 1))]
RUNS = 25

ort.set_default_logger_severity(3)
print(f"{'model':18}{'production':>22}{'Core AI (Python)':>18}{'speedup':>9}   accuracy vs ORT CPU (Core AI)   4-thread check")
for name, (lo, hi) in MODELS:
    onnx_path = f"{M}/{name}.onnx"
    ref_s = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feed = make_inputs(ref_s, lo, hi)
    ref = ref_s.run(None, feed)
    prod = create_session(onnx_path)
    label = prod.get_providers()[0].replace("ExecutionProvider", "") if hasattr(prod, "get_providers") else "MLX"
    cai = CoreAISession(f"{HERE}/out/{name}.aimodel", onnx_path)
    for _ in range(3):
        prod.run(None, feed), cai.run(None, feed)
    tp, tc = [], []
    for _ in range(RUNS):
        s = time.perf_counter(); prod.run(None, feed); tp.append(time.perf_counter() - s)
        s = time.perf_counter(); out = cai.run(None, feed); tc.append(time.perf_counter() - s)
    mp, mc = np.median(tp) * 1000, np.median(tc) * 1000
    acc = ", ".join(f"{score(o, r)[0]:.1f} dB" for o, r in zip(out, ref))

    # Concurrency: 4 threads x 5 runs on one session must all match a serial run.
    results, errors = [], []

    def worker(sess=cai):
        try:
            for _ in range(5):
                results.append(sess.run(None, feed)[0])
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    ts = [threading.Thread(target=worker) for _ in range(4)]
    s = time.perf_counter()
    [t.start() for t in ts]; [t.join() for t in ts]  # noqa: E702
    wall = (time.perf_counter() - s) * 1000 / 20
    same = all(np.array_equal(r, out[0]) for r in results)
    conc = f"{'OK' if same and not errors else 'MISMATCH' if not errors else 'ERROR'} ({len(results)}/20, {wall:.1f} ms/run)"
    print(f"{name:18}{f'{mp:.1f} ms ({label})':>22}{f'{mc:.1f} ms':>18}{mp / mc:>8.2f}x   {acc:30}  {conc}")
    del cai, prod

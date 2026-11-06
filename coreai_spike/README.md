# Core AI spike

Can Apple's Core AI (macOS 27) beat the current per-model backends?
Measured 2026-11-05 on an M5 Pro, macOS 27.0, Xcode 27.2, coreai-core 1.0.0b3,
coreai-onnx 1.2.1.

## Result

From Python through the ctypes bridge, all input/output copies included,
alternating runs with the production session (`e2e.py`):

| Model | Production | Core AI | Speedup | Accuracy vs ORT CPU |
|---|---|---|---|---|
| hyperswap_1a_256 | 22.8 ms (CoreML EP) | 17.2 ms | 1.33x | 80.3 dB (CoreML EP: 81.1) |
| GFPGANv1.4 | 144 ms (MLX interpreter) | 78 ms | 1.84x | 129.5 dB |
| GPEN-BFR-512 | 141 ms (MLX interpreter) | 106 ms | 1.33x | 123.3 dB |
| inswapper_128 | 32.9 ms (CoreML EP) | 35.4 ms | 0.93x | 119.8 dB |
| yoloface_8n | 2.7 ms (CoreML EP) | 3.8 ms (Swift-side) | 0.71x | 152 dB |

Four threads sharing one session gave bit-identical outputs.

## Findings

- Preferring `.neuralEngine`, `.gpu` or the default gave identical timings and
  outputs: the preference looks like a hint, not a pin.
- Preferring `.cpu` returns **wrong output** (hyperswap 25 dB, max error 1.2).
  Never route to it.
- First-ever load specializes the model (1-4 s); later loads hit Core AI's
  cache (~0.2-0.8 s).
- The bridge costs ~1.7 ms per call over Swift-side timing (hyperswap 15.5 ->
  17.2 ms).

## Files

- `CoreAIBridge.swift` - C ABI (`cai_load`, `cai_run`, `cai_free`) for ctypes.
- `coreai_session.py` - ORT-shaped `CoreAISession` over the bridge.
- `CoreAIBench.swift` - standalone Swift timing harness.
- `compare.py` - one model: Swift-side timings per compute unit vs production.
- `e2e.py` - the table above.

## Reproduce

```bash
# converter (now in requirements.txt; coreai-onnx is third-party, reviewed)
venv/bin/python -m coreai_onnx._cli convert models/hyperswap_1a_256.onnx \
    -o coreai_spike/out/hyperswap_1a_256.aimodel --compute-unit ane

# build (Xcode 27 SDK; CommandLineTools lacks CoreAI)
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
xcrun --sdk macosx swiftc -O -emit-library -target arm64-apple-macos27.0 \
    coreai_spike/CoreAIBridge.swift -o coreai_spike/out/libcoreai_bridge.dylib
xcrun --sdk macosx swiftc -O -parse-as-library -target arm64-apple-macos27.0 \
    coreai_spike/CoreAIBench.swift -o coreai_spike/out/coreai-bench

# run with the main repo venv
../roop-mlx/venv/bin/python coreai_spike/e2e.py
```

## Integrated

Shipped as `roop/coreai/` + `_COREAI_MODELS` in `roop/inference.py` (see
there for the final routing, including HyperSwap 1B/1C and GPEN 1024/2048).
Concurrent runs on one session corrupted GPEN-2048 output ~1 in 40, so the
shipped session serialises runs. AlphaFace and ReSwapper 256 measured slower.

## Open questions (resolved)

- Shipping: users need the Xcode 27 SDK to build the dylib (CommandLineTools
  has no CoreAI), or we ship a prebuilt one.
- Conversion: needs coreai-onnx (third-party) + coreai-core at install or
  first-use time; ~3-9 s per model.
- Core AI is beta; keep per-model parity tests as the gate, as with CoreML.

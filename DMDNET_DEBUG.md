# DMDNet Grid Artifact Debug State

## Status
BGR→RGB fix applied and working (colors are now correct).
Grid artifact remains — root cause identified, fix in progress.

## Root Cause
The grid originates in the encoder at the `f128` level (128×128 features), produced
by the dilated residual blocks `res3`/`res4` (dilation=3 at 128×128 resolution).

Stage-by-stage trace:
- `f256` (256×256): **clean, no grid**
- `f128` (128×128): **faint grid appears here** ← origin
- `f64` (64×64): grid is amplified
- MSDilate output: grid clearly visible

## What we know

### Spectral norm (conv weights)
- 248 out of all conv layers fail the `u @ W @ v` power-iteration computation
  (overflow to NaN/Inf in float64 due to very large weight values).
- These correctly fall back to SVD for the true spectral norm.
- After SVD normalisation all weights are finite (no NaN/Inf).
- BUT: the SVD sigma may differ slightly from the u/v approximation
  (e.g. MSDilate.conv1.0: sigma_uv=12.69 vs sigma_svd=12.71).
- This ~0.2% difference is unlikely to cause a visible grid.

### Memory attention
- Memory key magnitudes are large (le_64_mem_key std=14.6, max=83).
- With scaled dot-product this causes sim values of −10,000 to +4,000.
- Already switched to **cosine similarity** in `_read_mem`; sim now in [−1, 1].
- Despite the switch, grid persists → memory attention is NOT the primary cause.

### Next hypothesis to test
The overflow in `W_flat @ v_np` (float64) for layers like MSDilate.conv1.0
was interrupted before verification. Need to check:
1. What are the max raw weight values for those layers?
2. Does the SVD sigma differ enough from the true sigma to un-normalise the weights?
3. If yes, manually normalise those layers with the u/v approximation instead.

## Key files
- `roop/mlx_runtime/dmdnet.py` — full MLX implementation
  - `_read_mem()` line ~236: cosine similarity (already fixed)
  - `_preprocess_weights()` line ~799: spectral norm (SVD fallback)
- `roop/processors/Enhance_DMDNet.py` — BGR→RGB fix applied, scale_factor fixed

## Quick repro
```bash
source venv/bin/activate && python - <<'EOF'
import cv2, numpy as np
from roop.mlx_runtime.dmdnet import DMDNetMLX
import warnings; warnings.filterwarnings("ignore")
model = DMDNetMLX("models/DMDNet.pth")
img = cv2.imread("output/<any_output_file>")
img = cv2.resize(img, (512, 512))
frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
inp = frame_rgb.transpose(2, 0, 1)[np.newaxis]
locs = np.array([[(140,155,220,225),(290,155,370,225),(215,215,295,315),(175,325,335,395)]], dtype=np.float32)
out = model.forward(inp, locs)
out_hwc = out[0].transpose(1, 2, 0)
result = np.clip((out_hwc*0.5+0.5)*255, 0, 255).astype(np.uint8)
cv2.imwrite("/tmp/dmdnet_out.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
EOF
```

## Alternatives if grid can't be fixed
1. Skip DMDNet and use GFPGAN or CodeFormer instead (both work correctly).
2. Expose DMDNet as "experimental / known-broken" in the UI.
3. Apply a mild Gaussian blur to the DMDNet output before pasting to hide the grid.

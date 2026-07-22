"""Frame_Upscale -- MLX backend.

Supports RealESRGAN ×2 and SPAN ×4 super-resolution models.  Uses the
same tile-based pipeline as the upstream ORT version.
"""

import numpy as np

from roop.typing import Frame
from roop.utilities import resolve_relative_path

# (model file, scale, tile size).  SPAN uses 512px tiles: fewer tiles cut
# python tile-management overhead (110 -> 88 ms per 480p frame vs 128px)
# and reduce tile-border seams; the CoreML static pin in
# roop.inference._STATIC_DIM_SIZES must match.
_MODEL_CONFIGS = {
    "esrganx2": ("real_esrgan_x2.onnx",  2, 128),
    "spanx4":   ("span_kendata_x4.onnx", 4, 512),
}


class Frame_Upscale:
    plugin_options: dict = None
    prev_type: str = None

    processorname = "upscale"
    type = "frame_enhancer"

    def Initialize(self, plugin_options: dict):
        subtype = plugin_options.get("subtype", "spanx4")
        if self.prev_type is not None and self.prev_type != subtype:
            self.Release()

        self.plugin_options = plugin_options
        self.prev_type = subtype

        if getattr(self, "_session", None) is not None:
            return

        fname, self.scale, self.tile_size = _MODEL_CONFIGS.get(
            subtype, ("span_kendata_x4.onnx", 4, 512)
        )
        model_path = resolve_relative_path(f"../models/Frame/{fname}")

        from roop.inference import create_session
        self._session = create_session(model_path)
        self._output_name = self._session.get_outputs()[0].name

    def getProcessedResolution(self, width: int, height: int):
        return (width * self.scale, height * self.scale)

    def _prepare_tile(self, tile: Frame) -> np.ndarray:
        t = np.expand_dims(tile[:, :, ::-1], 0)  # BGR→RGB, add batch
        t = t.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        return t

    def _normalize_tile(self, tile: np.ndarray) -> Frame:
        t = tile.transpose(0, 2, 3, 1).squeeze(0) * 255.0
        return t.clip(0, 255).astype(np.uint8)[:, :, ::-1]  # RGB→BGR

    def _create_tiles(self, frame: Frame, size):
        tile_w, pad, overlap = size
        frame = np.pad(frame, ((pad, pad), (pad, pad), (0, 0)))
        inner_w = tile_w - 2 * overlap
        pad_bottom = overlap + inner_w - frame.shape[0] % inner_w
        pad_right  = overlap + inner_w - frame.shape[1] % inner_w
        frame = np.pad(frame, ((overlap, pad_bottom), (overlap, pad_right), (0, 0)))
        ph, pw = frame.shape[:2]
        tiles = []
        for r in range(overlap, ph - overlap, inner_w):
            for c in range(overlap, pw - overlap, inner_w):
                tiles.append(frame[r-overlap:r+overlap+inner_w,
                                   c-overlap:c+overlap+inner_w])
        return tiles, pw, ph

    def _merge_tiles(self, tiles, out_w, out_h, pad_w, pad_h, size):
        tile_w, pad, overlap = size
        merged = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
        inner_w = tiles[0].shape[1] - 2 * overlap
        cols = min(pad_w // inner_w, len(tiles))
        for i, tile in enumerate(tiles):
            tile = tile[overlap:-overlap, overlap:-overlap]
            r, c = divmod(i, cols)
            merged[r*tile.shape[0]:(r+1)*tile.shape[0],
                   c*tile.shape[1]:(c+1)*tile.shape[1]] = tile
        return merged[pad:pad+out_h, pad:pad+out_w]

    def Run(self, temp_frame: Frame) -> Frame:
        size = (self.tile_size, 8, 2)
        orig_h, orig_w = temp_frame.shape[:2]
        tiles, pad_w, pad_h = self._create_tiles(temp_frame, size)

        for i, tile in enumerate(tiles):
            inp = self._prepare_tile(tile)
            result = self._session.run([self._output_name], {"input": inp})[0]
            tiles[i] = self._normalize_tile(result)

        sc = self.scale
        out_size = (size[0]*sc, size[1]*sc, size[2]*sc)
        return self._merge_tiles(tiles, orig_w*sc, orig_h*sc,
                                 pad_w*sc, pad_h*sc, out_size)

    def Release(self):
        self._session = None

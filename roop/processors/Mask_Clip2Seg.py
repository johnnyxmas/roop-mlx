"""Mask_Clip2Seg -- MLX port pending.

Upstream used CLIPSeg (``clip/clipseg.py`` + ``rd64-uni-refined.pth``)
under PyTorch to generate text-prompted masks (e.g. "hair", "glasses").
"""


class Mask_Clip2Seg:
    plugin_options: dict = None

    processorname = "clip2seg"
    type = "mask"

    def Initialize(self, plugin_options: dict):
        raise NotImplementedError(
            "Mask_Clip2Seg has not been ported to MLX yet."
        )

    def Run(self, frame, masking_text: str):
        raise NotImplementedError(
            "Mask_Clip2Seg has not been ported to MLX yet."
        )

    def Release(self):
        return None

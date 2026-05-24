"""Enhance_DMDNet -- MLX port pending.

Upstream was a PyTorch implementation (full nn.Module definitions of
FeatureExtractor, MSDilateBlock, DMDNet, etc.) loaded from
``models/DMDNet.pth``.  The MLX port will need both the model
architecture re-implemented in ``mlx.nn`` and a weight converter that
maps the PyTorch state_dict keys to the new modules.

The component-location helpers (``landmarks106_to_68``,
``get_component_location``) are small NumPy/torch utilities that can
be brought back from git history once the model is ported.
"""


class Enhance_DMDNet:
    plugin_options: dict = None
    torchdevice = None

    processorname = "dmdnet"
    type = "enhance"

    def Initialize(self, plugin_options: dict):
        raise NotImplementedError(
            "Enhance_DMDNet has not been ported to MLX yet."
        )

    def Run(self, source_faceset, target_face, temp_frame):
        raise NotImplementedError(
            "Enhance_DMDNet has not been ported to MLX yet."
        )

    def Release(self):
        return None

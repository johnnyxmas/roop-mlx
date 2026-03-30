SWAP_MODEL_OUTPUT_SIZES = {
    "InSwapper 128 (CPU)": 128,
    "InSwapper 128 FP16": 128,
    "ReSwapper 128": 128,
    "ReSwapper 256": 256,
    "Ghost 1": 256,
    "Ghost 2": 256,
    "Ghost 3": 256,
    "HyperSwap 1A": 256,
    "HyperSwap 1B": 256,
    "HyperSwap 1C": 256,
    "UniFace": 256,
}

FF_SWAP_MODELS = {"Ghost 1", "Ghost 2", "Ghost 3", "HyperSwap 1A", "HyperSwap 1B", "HyperSwap 1C", "UniFace"}


class ProcessOptions:
    def __init__(
        self,
        swap_model,
        processordefines: dict,
        face_distance,
        blend_ratio,
        swap_mode,
        selected_index,
        masking_text,
        imagemask,
        num_steps,
        subsample_size,
        show_face_area,
        restore_original_mouth,
        show_mask=False,
    ):
        if swap_model is not None:
            self.swap_modelname = swap_model
            self.swap_output_size = SWAP_MODEL_OUTPUT_SIZES.get(swap_model, 128)
        else:
            self.swap_output_size = 128
        self.processors = processordefines
        self.face_distance_threshold = face_distance
        self.blend_ratio = blend_ratio
        self.swap_mode = swap_mode
        self.selected_index = selected_index
        self.masking_text = masking_text
        self.imagemask = imagemask
        self.num_swap_steps = num_steps
        self.show_face_area_overlay = show_face_area
        self.show_face_masking = show_mask
        self.subsample_size = subsample_size
        self.restore_original_mouth = restore_original_mouth
        self.max_num_reuse_frame = 15

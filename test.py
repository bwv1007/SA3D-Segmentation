from matanyone import InferenceCore
processor = InferenceCore("PeiqingYang/MatAnyone")
foreground_path, alpha_path = processor.process_video(
    input_path = "MA_inputs/video/images",
    mask_path = "MA_inputs/mask/00022.png",
    output_path = "MA_outputs"
)
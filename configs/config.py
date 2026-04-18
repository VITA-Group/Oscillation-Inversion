from dataclasses import dataclass

@dataclass
class Implicit2DNetworkConfig():
    hidden_dim: int
    num_layers: int

@dataclass
class DepthAlignConfig():
    prompt: str
    num_inference_steps: int
    num_predict_steps: int
    num_fixed_point_steps: int
    num_align_steps: int
    device: str
    flux_model_id: str
    path_fg_image: str
    path_bg_image: str
    path_target_image: str
    image_height: int
    image_width: int
    
    config_warping_net: Implicit2DNetworkConfig
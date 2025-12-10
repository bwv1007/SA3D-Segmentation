import os
from PIL import Image
import cv2
import torch
from tqdm import tqdm
from argparse import ArgumentParser

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

if __name__ == '__main__':
    
    parser = ArgumentParser(description="SAM feature extracting params")
    parser.add_argument("--image_root", default='data/fortress', type=str)
    parser.add_argument("--sam_checkpoint_path", default="./third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    parser.add_argument("--downscale", default=1, type=int)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    print("Initializing SAM...")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)


    
    IMAGE_DIR = os.path.join(args.image_root, 'images')
    
    #assert os.path.exists(IMAGE_DIR) and "Please specify a valid image root"
    OUTPUT_DIR = os.path.join(args.image_root, 'features2')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Extracting features...")
    for path in tqdm(os.listdir(IMAGE_DIR)):
        name = path.split('.')[0]
        img = cv2.imread(os.path.join(IMAGE_DIR, path))
        img = cv2.resize(img,dsize=(1024,1024),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
        predictor.set_image(img)
        features = predictor._features
        torch.save(features, os.path.join(OUTPUT_DIR, name+'.pt'))
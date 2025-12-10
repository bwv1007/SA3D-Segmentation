import os
from PIL import Image
import numpy as np
import argparse

def img2bin_mask(input_dir, output_dir, threshold=128):
    os.makedirs(output_dir, exist_ok=True)
    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    for fname in img_files:
        path = os.path.join(input_dir, fname)
        img = Image.open(path)
        arr = np.array(img)
        mask = (arr == threshold).astype(np.uint8) * 255
        out_img = Image.fromarray(mask)
        save_path = os.path.join(output_dir, fname)
        out_img.save(save_path)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图片转二值掩码")
    parser.add_argument('--input_dir', type=str, default="data/office3/remask", help='输入图片文件夹')
    parser.add_argument('--output_dir', type=str, default="data/office3/gtmask", help='输出二值掩码文件夹')
    parser.add_argument('--threshold', type=int, default=31, help='二值化阈值，默认128')
    args = parser.parse_args()
    img2bin_mask(args.input_dir, args.output_dir, args.threshold)

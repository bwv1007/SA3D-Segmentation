import os
from PIL import Image
import numpy as np
import argparse

def img2value_mask(input_dir, output_dir, value=255):
    os.makedirs(output_dir, exist_ok=True)
    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    for fname in img_files:
        path = os.path.join(input_dir, fname)
        img = Image.open(path)
        arr = np.array(img)
        # 只保留等于指定值的像素为前景，其余为背景
        mask = (arr == value).astype(np.uint8) * 255
        out_img = Image.fromarray(mask)
        save_path = os.path.join(output_dir, fname)
        out_img.save(save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按像素值生成二值掩码")
    parser.add_argument('--input_dir', type=str, required=True, help='输入图片文件夹')
    parser.add_argument('--output_dir', type=str, required=True, help='输出掩码文件夹')
    parser.add_argument('--value', type=int, default=255, help='指定前景像素值，默认255')
    args = parser.parse_args()
    img2value_mask(args.input_dir, args.output_dir, args.value)

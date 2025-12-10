import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(original_image_path, processed_image_path):
    """
    加载两张图片并计算PSNR和SSIM。
    """
    try:
        # 读取原始图片和处理后的图片
        gt_image = cv2.imread(original_image_path)
        upsampled_image = cv2.imread(processed_image_path)

        if gt_image is None or upsampled_image is None:
            raise FileNotFoundError("无法读取其中一张或两张图片。")

        # 确保两张图片的尺寸一致
        if gt_image.shape != upsampled_image.shape:
            # 如果尺寸不匹配，将上采样图片调整为与原图一致
            upsampled_image = cv2.resize(upsampled_image, (gt_image.shape[1], gt_image.shape[0]))
            print("注意: 图片尺寸不匹配，已将处理后的图片调整为与原图一致。")

        # --- 计算 PSNR ---
        psnr_value = cv2.PSNR(gt_image, upsampled_image)

        # --- 计算 SSIM ---
        # ssim函数要求图片为灰度图或者明确指定多通道
        # multichannel=True 适用于彩色图像
        (ssim_value, _) = ssim(gt_image, upsampled_image, full=True, multichannel=True, channel_axis=2)

        return psnr_value, ssim_value

    except FileNotFoundError as e:
        print(e)
        return None, None
    except Exception as e:
        print(f"计算指标时发生错误: {e}")
        return None, None

# --- 在您的主代码流程之后调用 ---
# 假设您的原始图片是 '1.jpg'，上采样后的图片是 'upsampled_to_original.png'
original_file = 'test1.jpg'
upsampled_file = 'test2.jpg'

# 首先运行您提供的代码来生成 'upsampled_to_original.png'
# ... (您提供的完整代码) ...

# 然后计算指标
psnr, ssim_metric = calculate_metrics(original_file, upsampled_file)

if psnr is not None and ssim_metric is not None:
    print("\n--- 图像质量评估结果 ---")
    print(f"PSNR (峰值信噪比): {psnr:.2f} dB")
    print(f"SSIM (结构相似性): {ssim_metric:.4f}")
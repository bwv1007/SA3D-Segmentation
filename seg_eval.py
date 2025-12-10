import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse


def evaluate_masks(gt_mask_folder, pred_mask_folder, comparison_folder=None):
    """
    Computes segmentation metrics and saves visual comparisons for each mask pair.

    Args:
        gt_mask_folder (str): Path to the folder containing ground-truth masks.
        pred_mask_folder (str): Path to the folder containing predicted (rendered) masks.
        comparison_folder (str, optional): Path to save comparison images. If None, no images are saved.
    """
    # 1. 获取并排序文件列表
    print("Searching for mask files and matching them by sorted order...")
    gt_files = sorted([f for f in os.listdir(gt_mask_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    pred_files = sorted([f for f in os.listdir(pred_mask_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # 检查文件数量
    if len(gt_files) != len(pred_files):
        print("\n" + "=" * 50)
        print(" Critical Error: File count mismatch!")
        print(f" Ground-Truth Folder ('{os.path.basename(gt_mask_folder)}') contains: {len(gt_files)} images")
        print(f" Prediction Folder ('{os.path.basename(pred_mask_folder)}') contains: {len(pred_files)} images")
        print(" Cannot proceed with order-based matching. Please check the folders.")
        print("=" * 50)
        return
    if not gt_files:
        print("\nError: No mask files found to evaluate.")
        return

    print(f"Found {len(gt_files)} mask pairs. Proceeding with evaluation.")
    file_pairs = list(zip(gt_files, pred_files))

    ### <<< 新增代码开始 >>> ###
    # 2. 如果提供了保存路径，则创建文件夹
    if comparison_folder:
        os.makedirs(comparison_folder, exist_ok=True)
        print(f"Comparison images will be saved to: '{comparison_folder}'")
    ### <<< 新增代码结束 >>> ###

    # 3. 初始化指标累加器
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    # 4. 逐对加载、处理和评估
    for gt_filename, pred_filename in tqdm(file_pairs, desc="Evaluating mask pairs"):
        gt_path = os.path.join(gt_mask_folder, gt_filename)
        pred_path = os.path.join(pred_mask_folder, pred_filename)

        # 加载并二值化 Ground-Truth Mask
        gt_mask_raw = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask_raw is None:
            print(f"Warning: Could not read GT mask {gt_filename}, skipping.")
            continue
        gt_mask = (gt_mask_raw > 127).astype(np.uint8)  # 阈值设为127更通用

        # 加载并二值化 Prediction Mask
        pred_mask_raw = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if pred_mask_raw is None:
            print(f"Warning: Could not read predicted mask {pred_filename}, skipping.")
            continue
        pred_mask = (pred_mask_raw > 127).astype(np.uint8)

        # 统一尺寸
        if gt_mask.shape != pred_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 累加像素指标
        tp = np.sum(np.logical_and(gt_mask, pred_mask))
        fp = np.sum(np.logical_and(np.logical_not(gt_mask), pred_mask))
        fn = np.sum(np.logical_and(gt_mask, np.logical_not(pred_mask)))
        tn = np.sum(np.logical_and(np.logical_not(gt_mask), np.logical_not(pred_mask)))

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        ### <<< 新增代码开始 >>> ###
        # 5. 如果需要，生成并保存对比图
        if comparison_folder:
            # 创建一个三通道的彩色画布用于差异图
            h, w = gt_mask.shape
            diff_map = np.zeros((h, w, 3), dtype=np.uint8)

            # BGR color format for OpenCV
            # True Positive (White)
            diff_map[np.logical_and(gt_mask == 1, pred_mask == 1)] = [255, 255, 255]
            # False Positive (Red)
            diff_map[np.logical_and(gt_mask == 0, pred_mask == 1)] = [0, 0, 255]
            # False Negative (Blue)
            diff_map[np.logical_and(gt_mask == 1, pred_mask == 0)] = [255, 0, 0]

            # 将单通道的GT和Pred掩码转换为三通道B&W图像以便拼接
            gt_mask_bgr = cv2.cvtColor(gt_mask * 255, cv2.COLOR_GRAY2BGR)
            pred_mask_bgr = cv2.cvtColor(pred_mask * 255, cv2.COLOR_GRAY2BGR)

            # 水平拼接三张图：[GT | Pred | Diff]
            comparison_image = np.hstack((gt_mask_bgr, pred_mask_bgr, diff_map))

            # 保存图像
            save_path = os.path.join(comparison_folder, gt_filename)
            cv2.imwrite(save_path, comparison_image)
        ### <<< 新增代码结束 >>> ###

    # 6. 最终指标计算与打印
    print("\n" + "=" * 50)
    print("           Final Segmentation Evaluation Results")
    print("=" * 50)
    if len(file_pairs) > 0:
        denominator_iou = total_tp + total_fp + total_fn
        final_mIoU = total_tp / denominator_iou if denominator_iou > 0 else 0
        print(f"Mean Intersection over Union (mIoU): {final_mIoU:.4f}")

        denominator_acc = total_tp + total_fp + total_fn + total_tn
        final_mAcc = (total_tp + total_tn) / denominator_acc if denominator_acc > 0 else 0
        print(f"Mean Pixel Accuracy (mAcc):           {final_mAcc:.4f}")

        # --- 新增：精确率和召回率 ---
        denominator_precision = total_tp + total_fp
        final_precision = total_tp / denominator_precision if denominator_precision > 0 else 0
        print(f"Precision:                            {final_precision:.4f}")

        denominator_recall = total_tp + total_fn
        final_recall = total_tp / denominator_recall if denominator_recall > 0 else 0
        print(f"Recall:                               {final_recall:.4f}")

        print("\n--- Aggregated Pixel Counts ---")
        print(f"Total True Positives (TP): {total_tp:,}")
        print(f"Total True Negatives (TN): {total_tn:,}")
        print(f"Total False Positives (FP): {total_fp:,}")
        print(f"Total False Negatives (FN): {total_fn:,}")
    else:
        print("Evaluation finished, but no files were processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation masks and optionally save visual comparisons.",
        formatter_class=argparse.RawTextHelpFormatter   # 格式化帮助信息，使其更易读
    )

    parser.add_argument("--gt_folder", type=str, default="data/fortress/gtmask",
                        help="Path to the folder containing the ground-truth masks.")

    parser.add_argument("--pred_folder", type=str,  default="data/fortress/turemask",
                        help="Path to the folder containing the predicted (rendered) masks.")

    ### <<< 新增代码开始 >>> ###
    parser.add_argument("--comparison_folder", type=str, default="data/fortress/comparison",
                        help="""(Optional) Path to the folder where comparison images will be saved.
If not provided, no images will be saved.
Each saved image contains three parts: [GroundTruth | Prediction | Difference Map]
Difference Map Colors:
- White: True Positive (Correctly predicted foreground)
- Red:   False Positive (Wrongly predicted as foreground)
- Blue:  False Negative (Missed foreground)
- Black: True Negative (Correctly predicted background)""")
    ### <<< 新增代码结束 >>> ###

    args = parser.parse_args()

    evaluate_masks(args.gt_folder, args.pred_folder, args.comparison_folder)
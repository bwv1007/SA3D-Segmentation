# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 程序可见的GPU
import torch
from scene import Scene, GaussianModel
import json
from tqdm import tqdm
from os import makedirs
# from gaussian_renderer import render, render_feature, render_contrastive_feature, render_xyz
from gaussian_renderer import render
from gaussian_renderer import flashsplat_render
import torchvision
import copy
import cv2
import argparse
import numpy as np
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams

# MASK_THRESHOLD = 0.1

mask_path = ''


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, target, MASK_THRESHOLD,
               gt_mask_folder, pre_seg_gaussians=None):
    #'''''''''''mask置信初始化参数 '''''''''''
    gt_mask_therhold = 0.9
    sa3d_mask_therhold = 2
    finall_mask_therhold = 0
    # '''''''''''splat初始化参数 '''''''''''
    view_num = -1  # 决定了我们从整个数据集中采样多少张图片（及其对应的掩码）来“投票”决定最终的3D分割结果
    all_counts = None  # 所有贡献度的张量
    obj_num = 1  # 模型中分割类别的数量
    slackness = -0.4  # 背景偏置度
    obj_id = -1
    ref_weight = 0  # 增大第一帧的权重
    # '''''''''''splat初始化参数 '''''''''''
    render_path = os.path.join(model_path, name, f"ours_{iteration}_{str(MASK_THRESHOLD).replace('.', '_')}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}_{str(MASK_THRESHOLD).replace('.', '_')}", "gt")
    global mask_path
    mask_path = os.path.join(model_path, name, f"ours_{iteration}_{str(MASK_THRESHOLD).replace('.', '_')}", "mask")
    sa3d_mask_path = os.path.join(model_path, name, f"ours_{iteration}_{str(MASK_THRESHOLD).replace('.', '_')}",
                                  "sa3d_mask")
    splat_mask_path = os.path.join(model_path, name, f"ours_{iteration}_{str(MASK_THRESHOLD).replace('.', '_')}",
                                   "splat_mask")
    masks_dir = gt_mask_folder
    num_views = len(views)
    all_true_gt_masks = []
    splat_mask = [None] * num_views
    all_files = os.listdir(masks_dir)
    pre_render_model = pre_seg_gaussians if pre_seg_gaussians is not None else gaussians
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    mask_filenames = sorted([f for f in all_files if f.lower().endswith(image_extensions)])
    if len(mask_filenames) < num_views:
        raise FileNotFoundError(
            f"Found {len(mask_filenames)} masks in '{masks_dir}', but {num_views} views require masks."
        )
    mask_filenames = mask_filenames[-num_views:]
    print(f"Found {len(mask_filenames)} images in '{masks_dir}'. Loading...")
    for mask_filename in mask_filenames:
        # mask_filename = f"{i:05d}.png"
        # mask_filename = f"image{i:03d}_pseudo.png"
        gt_mask_path = os.path.join(masks_dir, mask_filename)
        gt_mask_image = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask_image is None:
            raise FileNotFoundError(f"Could not load GT mask for evaluation: {gt_mask_path}")
        gt_mask_tensor = torch.from_numpy(gt_mask_image).float().cuda()
        all_true_gt_masks.append(gt_mask_tensor / 255.)

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(mask_path, exist_ok=True)
    makedirs(sa3d_mask_path, exist_ok=True)
    makedirs(splat_mask_path, exist_ok=True)
    tmp_vis_path = "tmpvis_files/orchids"
    history_filename = os.path.join(tmp_vis_path, "iou_scores.json")
    with open(history_filename, 'r') as f:
        iou_data = json.load(f)
    # if target == 'feature':
    #     render_func = render_feature
    # elif target == 'contrastive_feature':
    #     render_func = render_contrastive_feature
    # elif target == 'xyz':
    #     render_func = render_xyz
    render_func = render

    if view_num > 0:
        view_idx = np.linspace(0, len(views) - 1, view_num, dtype=int).tolist()  # 0, len(views) - 1中分割出view_num个数
        views_used = [views[idx] for idx in view_idx]
        # views_used = [views[idx] for idx in view_idx]
    else:
        view_num = len(views)
        views_used = views

    stats_counts_path = os.path.join(model_path, name, f"ours_{iteration}_{str(MASK_THRESHOLD).replace('.', '_')}",
                                     "used_count")
    makedirs(stats_counts_path, exist_ok=True)
    cur_count_dir = os.path.join(stats_counts_path, "view_num_{:03d}_objnum_{:03d}.pth".format(view_num, obj_num))
    # if os.path.exists(cur_count_dir):
    #     print(f"find {cur_count_dir}")
    #     all_counts = torch.load(cur_count_dir).cuda()
    # else:
    all_counts = None
    for idx, view in enumerate(tqdm(views_used, desc="Rendering progress")):
        base_view_idx = view.uid
        #base_view_idx = view.uid + iteration - view_num  # 三轮迭代+22    总数-view个数+1
        if obj_num == 1:
            # 加载gtmask
            # gt_mask = view.objects.to(torch.float32) / 255.
            gt_mask = all_true_gt_masks[base_view_idx]
            gt_mask[gt_mask <= gt_mask_therhold] = 0
            gt_mask[gt_mask != 0] = 1
        else:
            gt_mask = view.objects.to(torch.float32)
            assert torch.any(torch.max(gt_mask) < obj_num), f"max_obj {int(torch.max(gt_mask).item())}"
        # set(tuple(gt_mask.cpu().tolist())[0])
        render_pkg = flashsplat_render(view, gaussians, pipeline, background, gt_mask=gt_mask, obj_num=obj_num)
        rendering = render_pkg["render"]
        used_count = render_pkg["used_count"]
        if all_counts is None:
            all_counts = torch.zeros_like(used_count)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        # all_counts += used_count * iou_data[base_view_idx]
        all_counts += used_count
        if view.uid == 0:
            all_counts += used_count * ref_weight
    # save_mp4(render_path, 15)
    torch.save(all_counts, cur_count_dir)

    if all_counts is not None:
        if obj_num == 1:
            # for binary seg,
            all_counts = torch.nn.functional.normalize(all_counts, dim=0)
            all_counts[0, :] += slackness  # slackness
            unique_label = all_counts.max(dim=0)[1]
        # else:
        # all_obj_labels = multi_instance_opt(all_counts, slackness)
        render_num = all_counts.size(0)
        # '''''用于渲染视频''''''
        _view_num = 10  # view_num only used to render the object
        view_idx = np.linspace(0, len(views) - 1, view_num, dtype=int).tolist()
        views_used = [views[idx] for idx in view_idx]

        early_stop = False
        for obj_idx in range(render_num):
            if obj_id > 0:
                obj_idx = obj_id
                early_stop = True
            if obj_num == 1:
                # 过滤，只保留标签为obj_idx的高斯点
                obj_used_mask = (unique_label == obj_idx)
            '''
            else:
                obj_used_mask = (all_obj_labels[obj_idx]).bool()
                if obj_used_mask.sum().item() == 0:
                    continue
                # obj_used_mask = ~obj_used_mask
            '''
            obj_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "obj_{:03d}".format(obj_idx))
            os.makedirs(obj_render_path, exist_ok=True)
            # ''''''''''''渲染splat_mask'''''''''
            for idx, view in enumerate(tqdm(views, desc="Rendering object {:03d}".format(obj_idx))):
                render_pkg = flashsplat_render(view, gaussians, pipeline, background, used_mask=obj_used_mask)
                # render_pkg = render(view, gaussians, pipeline, background, used_mask=obj_used_mask)
                '''
                rendering = render_pkg["render"]
                gt = view.original_image[0:3, :, :]
                torchvision.utils.save_image(rendering, os.path.join(obj_render_path, '{0:05d}'.format(idx) + ".png"))
                torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
                '''
                # mask = render_pkg["mask"]
                mask = render_pkg["alpha"]
                torchvision.utils.save_image(mask, os.path.join(splat_mask_path, '{0:05d}'.format(idx) + ".png"))
                if obj_num == 1:
                    splat_mask[view.uid] = mask
                # gt_mask = (view.objects / 255.).bool()[None, ].repeat(3, 1, 1)
                # dummy_gt_mask = torch.zeros_like(gt)
                # dummy_gt_mask[0] = 1.
                # gt[gt_mask.bool()] = 0.7 * gt[gt_mask.bool()] + 0.3 * dummy_gt_mask[gt_mask.bool()]

            # save_mp4(obj_render_path, 15)
            if early_stop:
                break

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        res = render_func(view, pre_render_model, pipeline, background)
        rendering = res["render"]

        gt = view.original_image[0:3, :, :]
        # print(rendering.shape, mask.shape, gt.shape, "rendering.shape, mask.shape, gt.shape")

        # print("mask render time", time.time() - start_time)
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image((res["depth"] - res["depth"].min()) / (res["depth"].max() - res["depth"].min()),
                                     os.path.join(render_path, '{0:05d}_depth'.format(idx) + ".png"))
        if target == 'seg' or target == 'scene':
            mask = res["mask"]
            mask[mask <= sa3d_mask_therhold] = 0.
            mask[mask > sa3d_mask_therhold] = 1.
            mask = mask[0, :, :]
            torchvision.utils.save_image(mask, os.path.join(sa3d_mask_path, '{0:05d}'.format(idx) + ".png"))
            _mask = mask * splat_mask[view.uid]
            # _mask = mask
            _mask[_mask <= finall_mask_therhold] = 0.
            _mask[_mask > 0] = 1.
            torchvision.utils.save_image(_mask, os.path.join(mask_path, f"{view.image_name}.png"))
        if target == 'seg' or target == 'scene' or target == 'coarse_seg_everything':
            _rendering = _mask * rendering
            # _rendering = rendering
            torchvision.utils.save_image(_rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            # torchvision.utils.save_image(gt * mask[None], os.path.join(render_path, f"{view.image_name}.png"))
        elif 'feature' in target:
            torch.save(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".pt"))
        elif target == 'xyz':
            torch.save(rendering, os.path.join(render_path, 'xyz_{0:05d}'.format(idx) + ".pt"))


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                segment: bool = False, target='scene', idx=0, precomputed_mask=None, MASK_THRESHOLD=0.1,
                gt_mask_folder=None):
    dataset.need_features = dataset.need_masks = False
    if segment:
        '''
        assert target == 'seg' or target == 'coarse_seg_everything' or precomputed_mask is not None and "Segmentation only works with target seg!"
        '''
        # 允许 --segment 与 --target scene 或 --target seg 组合使用
        allowed_targets = ['scene', 'seg', 'coarse_seg_everything']
        assert target in allowed_targets or precomputed_mask is not None, \
            f"When using --segment, the target must be one of {allowed_targets}, but got '{target}'."

    gaussians, feature_gaussians = None, None
    with torch.no_grad():
        if target == 'scene' or target == 'seg' or target == 'coarse_seg_everything' or target == 'xyz':
            gaussians = GaussianModel(dataset.sh_degree)
        # if target == 'feature' or target == 'coarse_seg_everything' or target == 'contrastive_feature':
        #    feature_gaussians = FeatureGaussianModel(dataset.feature_dim)

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,
                      init_from_3dgs_pcd=dataset.init_from_3dgs_pcd, mode='eval',
                      target=target if target != 'xyz' else 'scene')
        scene.save(scene.loaded_iter, target='scene', colored=True)
        pre_seg_gaussians=None
        pre_seg_gaussians = copy.deepcopy(gaussians)
        if segment:
            if target == 'coarse_seg_everything':
                mask = feature_gaussians.segment(idx=idx)
                gaussians.segment(mask=mask)
                scene.save(scene.loaded_iter, target=f'seg_res_{idx}')
            else:
                if precomputed_mask is None:
                    gaussians.segment()
                    scene.save(scene.loaded_iter, target='seg_res', colored=True)
                    scene.save(scene.loaded_iter, target='seg_wo')
                else:
                    pre_mask = torch.load(precomputed_mask)
                    gaussians.segment(pre_mask)
                    scene.save(scene.loaded_iter, target='seg_res')



        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        if 'feature' in target:
            gaussians = feature_gaussians
            bg_color = [1 for i in range(dataset.feature_dim)] if dataset.white_background else [0 for i in range(
                dataset.feature_dim)]

        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline,
                       background, target, MASK_THRESHOLD, gt_mask_folder, pre_seg_gaussians=pre_seg_gaussians)

        # if not skip_test:
        #     render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, target, MASK_THRESHOLD)


def resize_masks_in_folder(input_folder, output_folder, target_width=1600, target_height=1200):
    """
    Resizes and converts all mask images in a folder to a specific target size
    and ensures they are single-channel grayscale.

    Args:
        input_folder (str): Path to the folder containing original mask images.
        output_folder (str): Path to the folder where resized masks will be saved.
        target_width (int): The target width for the output images.
        target_height (int): The target height for the output images.
    """

    os.makedirs(output_folder, exist_ok=True)
    print(f"Output will be saved to: '{output_folder}'")

    image_filenames = [f for f in os.listdir(input_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
                       # and '_depth' in f
                       # and os.path.splitext(f)[0].isdigit()
                       ]

    if not image_filenames:
        print(f"Warning: No images found in the input folder: {input_folder}")
        return

    print(f"Found {len(image_filenames)} mask(s) to process.")

    # --- 2. 遍历并处理每一张图片 ---
    for filename in tqdm(image_filenames, desc="Processing masks"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # =========================================================================
        #         <<<--- 核心修复：使用更健壮的方式加载和转换 ---<<<
        # =========================================================================
        # 1. 以“原样”模式读取图像，保留所有通道 (例如 Alpha 通道)
        mask_raw = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

        if mask_raw is None:
            print(f"Warning: Could not read file {input_path}, skipping.")
            continue

        # 2. 从可能的多通道图像中，创建一个单通道的图像
        #    如果图像是3维的 (例如 H, W, 3)，我们检查任何一个通道是否>0
        if mask_raw.ndim == 3:
            # 对于(H, W, C)图像，在通道维度上取最大值。
            # 只要任何一个通道的值不是0，结果就不是0。
            # 这能正确处理(255,255,255)的白色和(255,0,0)的红色等情况。

            mask_single_channel = mask_raw.max(axis=2)
        else:
            # 如果已经是单通道的灰度图，直接使用
            mask_single_channel = mask_raw
        _, binary_mask = cv2.threshold(mask_single_channel, 1, 255, cv2.THRESH_BINARY)
        # =========================================================================

        # --- 3. 核心操作：对单通道图像进行缩放 ---
        # 现在，`mask_single_channel`可以保证是一个二维Numpy数组
        resized_mask = cv2.resize(
            binary_mask,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST
        )

        # 4. 保存处理后的单通道图像
        cv2.imwrite(output_path, resized_mask)

    print("\nMask processing complete!")


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
    for i, (gt_filename, pred_filename) in enumerate(tqdm(file_pairs, desc="Evaluating mask pairs")):
        if i == 14:  # 列表索引从0开始，因此第15个文件对的索引是14
            print(f"\nSkipping 15th pair as requested: GT='{gt_filename}', PRED='{pred_filename}'")
            continue
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
            gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

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
        ### <<< 新增代码结束 >>>

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
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    op = OptimizationParams(parser)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=94, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--segment", action="store_true")
    parser.add_argument('--target', default='scene', const='scene', nargs='?',
                        choices=['scene', 'seg', 'feature', 'coarse_seg_everything', 'contrastive_feature', 'xyz'])
    parser.add_argument('--idx', default=0, type=int)
    parser.add_argument('--precomputed_mask', default=None, type=str)
    parser.add_argument('--MASK_THRESHOLD', default=0.1, type=float)

    parser.add_argument("--output_folder", type=str, default="data/orchids/turemask",
                        help="Path to the folder where resized masks will be saved.")
    parser.add_argument("--width", type=int, default=1600,
                        help="Target width for the resized masks (default: 1600).")
    parser.add_argument("--height", type=int, default=1200,
                        help="Target height for the resized masks (default: 1200).")

    parser.add_argument("--gt_folder", type=str, default="data/orchids/gtmask",
                        help="Path to the folder containing the ground-truth masks.")
    parser.add_argument("--comparison_folder", type=str, default="data/orchids/comparison")
    parser.add_argument("--sam_folder", type=str, default="tmpvis_files/orchids/sam_masks",
    # parser.add_argument("--sam_folder", type=str, default="MA_outputs/images/pha",
                        help="Path to the folder containing the ground-truth masks.")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    if not hasattr(args, 'precomputed_mask'):
        args.precomputed_mask = None
    if args.precomputed_mask is not None:
        print("Using precomputed mask " + args.precomputed_mask)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.segment, args.target, args.idx, args.precomputed_mask, args.MASK_THRESHOLD, args.sam_folder)
    resize_masks_in_folder(mask_path, args.output_folder, args.width, args.height)
    resize_masks_in_folder(args.gt_folder, args.gt_folder, args.width, args.height)
    evaluate_masks(args.gt_folder, args.output_folder, args.comparison_folder)

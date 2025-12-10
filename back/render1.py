#
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
os.environ['CUDA_VISIBLE_DEVICES']='2' # 程序可见的GPU
import os
import torch
from scene import Scene, GaussianModel  # <--- 修改1：移除了 FeatureGaussianModel
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render  # <--- 修改2：只保留了标准的render
import torchvision

from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, target, MASK_THRESHOLD):
    # --- 这部分函数基本不变，但内部逻辑被简化 ---
    render_path = os.path.join(model_path, name, f"ours_{iteration}_{str(MASK_THRESHOLD).replace('.', '_')}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}_{str(MASK_THRESHOLD).replace('.', '_')}", "gt")
    mask_path = os.path.join(model_path, name, f"ours_{iteration}_{str(MASK_THRESHOLD).replace('.', '_')}", "mask")


    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(mask_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # 总是使用标准的render函数
        res = render(view, gaussians, pipeline, background)
        rendering = res["render"]

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        # 深度图渲染
        torchvision.utils.save_image((res["depth"] - res["depth"].min()) / (res["depth"].max() - res["depth"].min()),
                                     os.path.join(render_path, '{0:05d}_depth'.format(idx) + ".png"))

        # 如果目标是分割，保存掩码
        if target == 'seg':
            #深度置信
            min_depth = res["depth"].min()
            max_depth = res["depth"].max()
            DEPTH_SEG_THRESHOLD=0.8
            absolute_threshold = min_depth + (max_depth - min_depth) * DEPTH_SEG_THRESHOLD
            depth_mask = (res["depth"]<= absolute_threshold).float()
            #mask置信
            mask = res["mask"]
            mask[mask <= MASK_THRESHOLD] = 0.
            mask[mask > MASK_THRESHOLD] = 1.
            mask = mask[0, :, :]
            #mask与深度置信融合
            mask = mask * depth_mask
            torchvision.utils.save_image(mask, os.path.join(mask_path, f"{view.image_name}.png"))

        # RGB渲染结果
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                segment: bool = False, target='scene', precomputed_mask=None, MASK_THRESHOLD=0.1):
    # --- 修改3：简化了整个函数的逻辑 ---
    dataset.need_features = dataset.need_masks = False

    # 断言现在只检查'seg'
    if segment:
        assert target == 'seg' and "Segmentation only works with target 'seg'!"

    with torch.no_grad():
        # 总是初始化 GaussianModel
        gaussians = GaussianModel(dataset.sh_degree)

        # Scene的初始化不再需要 feature_gaussians
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, mode='eval', target=target)

        if segment:
            if precomputed_mask is None:
                # 默认的分割，直接操作gaussians
                gaussians.segment()
                scene.save(scene.loaded_iter, target='seg_res', colored=True)
            else:
                # 从预计算的掩码分割
                pre_mask = torch.load(precomputed_mask)
                gaussians.segment(pre_mask)
                scene.save(scene.loaded_iter, target='seg_res')

        # 背景颜色现在总是RGB
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline,
                       background, target, MASK_THRESHOLD)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline,
                       background, target, MASK_THRESHOLD)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=39, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--segment", action="store_true", help="Enable segmentation rendering mode.")

    # --- 修改4：简化了 target 的选项 ---
    parser.add_argument('--target', default='scene', const='scene', nargs='?',
                        choices=['scene', 'seg'],
                        help="Choose rendering target: 'scene' for RGB, 'seg' for segmentation mask.")

    parser.add_argument('--precomputed_mask', default=None, type=str,
                        help="Path to a precomputed mask file for segmentation.")
    parser.add_argument('--MASK_THRESHOLD', default=0.5, type=float, help="Threshold to binarize the rendered mask.")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # 兼容性检查
    if not hasattr(args, 'precomputed_mask'):
        args.precomputed_mask = None
    if args.precomputed_mask is not None:
        print("Using precomputed mask: " + args.precomputed_mask)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # 因为render_sets的签名改变了，我们不再需要传递idx
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                args.segment, args.target, args.precomputed_mask, args.MASK_THRESHOLD)
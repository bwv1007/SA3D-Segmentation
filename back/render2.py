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

import torch
from scene import Scene, GaussianModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision

from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args


# render_set 函数保持不变
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, target, MASK_THRESHOLD,
               segment_applied):
    mode = "rgb_renders"
    if target == 'seg':
        mode = f"masks_thresh_{str(MASK_THRESHOLD).replace('.', '_')}"
    elif segment_applied:
        mode = "cutouts"

    render_path = os.path.join(model_path, name, f"ours_{iteration}", mode)
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name} [{mode}]")):
        res = render(view, gaussians, pipeline, background)
        rendering = res["render"]

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{view.image_name}.png"))

        if target == 'seg':
            mask = (res["mask"] > MASK_THRESHOLD).float()
            torchvision.utils.save_image(mask, os.path.join(render_path, f"{view.image_name}.png"))
        else:
            torchvision.utils.save_image(rendering, os.path.join(render_path, f"{view.image_name}.png"))


# ===================== 核心修改在这里 =====================
def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                segment: bool, target: str, MASK_THRESHOLD: float):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)

        # --- 核心修复：决定加载哪个模型文件 ---
        # 如果用户想渲染分割掩码(seg)或者分割后的抠图(scene + segment),
        # 我们必须加载那个经过分割训练的模型, 也就是 seg_point_cloud.ply
        if target == 'seg' or (target == 'scene' and segment):
            load_target = 'seg'
        else:
            # 否则，加载原始的场景模型
            load_target = 'scene'

        print(f"Attempting to load model with target type: '{load_target}'...")
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, mode='eval', target=load_target)
        print(f"Successfully loaded 3D-GS model from iteration {scene.loaded_iter}.")

        segment_applied = False
        # 如果是渲染抠图，我们在加载了seg模型后，再执行一次segment()来移除背景
        # segment()会利用已经加载进来的 mask_confidence 来修改 opacity
        if segment and target == 'scene':
            print("Applying segmentation to create a cutout render...")
            gaussians.segment()
            segment_applied = True
            print("Segmentation applied for cutout.")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 传递 segment_applied 用于生成正确的文件夹名
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline,
                       background, target, MASK_THRESHOLD, segment_applied)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline,
                       background, target, MASK_THRESHOLD, segment_applied)


# ===================== 修改结束 =====================

if __name__ == "__main__":
    # (这部分和我们之前确认可用的版本完全一样)
    parser = ArgumentParser(description="Rendering script for 3D Gaussian Splatting models.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--segment", action="store_true",
                        help="Apply segmentation to the model before rendering RGB images to get a 'cutout' effect. Implicitly active for '--target seg'.")

    parser.add_argument('--target', default='scene', const='scene', nargs='?',
                        choices=['scene', 'seg'],
                        help="Specifies what to render: 'scene' for RGB, 'seg' for masks.")

    parser.add_argument('--MASK_THRESHOLD', default=1, type=float,
                        help="Threshold to binarize the rendered mask. Only for '--target seg'.")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    # render_sets 现在少了一个我们用不到的precomputed_mask参数
    render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                args.skip_train, args.skip_test, args.segment, args.target, args.MASK_THRESHOLD)
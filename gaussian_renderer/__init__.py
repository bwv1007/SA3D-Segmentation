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

import math
import os
import sys
import warnings

import torch
from diff_gaussian_rasterization_depth import (
    GaussianRasterizationSettings as DepthGaussianRasterizationSettings,
    GaussianRasterizer as DepthGaussianRasterizer,
)
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


from flashsplat_rasterization import GaussianRasterizationSettings as FlashSplat_GaussianRasterizationSettings
from flashsplat_rasterization import GaussianRasterizer as FlashSplat_GaussianRasterizer
import pdb
# import time



from cobgs_diff_gaussian_rasterization import (  # type: ignore
    GaussianRasterizationSettings as COB_GaussianRasterizationSettings,
    GaussianRasterizer as COB_GaussianRasterizer,
)
_COB_RASTERIZER_AVAILABLE = True


if not _COB_RASTERIZER_AVAILABLE:
    warnings.warn(
        "COB diff_gaussian_rasterization package not found. Falling back to depth-only rasterizer without mask signals.",
        RuntimeWarning,
    )

def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    override_color=None,
    override_mask=None,
    filtered_mask=None,
    opt=None,
):
    """Render the scene and optionally gather COB mask signals."""

    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz,
            dtype=pc.get_xyz.dtype,
            device=pc.get_xyz.device,
            requires_grad=True,
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except RuntimeError:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    mask = pc.get_mask if override_mask is None else override_mask
    if filtered_mask is not None:
        new_opacity = opacity.detach().clone()
        new_opacity[filtered_mask, :] = -1.0
        opacity = new_opacity

    include_mask = True if opt is None else getattr(opt, "include_mask", True)
    antialiasing = getattr(pipe, "antialiasing", False)

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = (
                pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    mask_signals = None

    raster_settings = COB_GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=antialiasing,
        include_mask=include_mask,
    )

    rasterizer = COB_GaussianRasterizer(raster_settings=raster_settings)
    if include_mask:
        mask_precomp = mask
        if mask_precomp is not None and mask_precomp.dim() == 2 and mask_precomp.shape[1] == 1:
            mask_precomp = mask_precomp.squeeze(1)
        mask_signals = torch.zeros((pc.get_xyz.shape[0], 2), requires_grad=True, device="cuda") + 0
        try:
            mask_signals.retain_grad()
        except RuntimeError:
            pass
    else:
        mask_precomp = torch.zeros((1,), dtype=opacity.dtype, device=opacity.device)
        mask_signals = torch.zeros((1,), device=opacity.device)

    rendered_image, rendered_mask, radii, invdepth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        mask_precomp=mask_precomp,
        mask_signals=mask_signals,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    rendered_depth = invdepth

    return {
        "render": rendered_image,
        "mask": rendered_mask,
        "depth": rendered_depth,
        "mask_signals": mask_signals,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }

# def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, filtered_mask = None):
#     """
#     Render the scene. 
    
#     Background tensor (bg_color) must be on GPU!
#     """
 
#     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
#     screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     # Set up rasterization configuration
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         sh_degree=pc.active_sh_degree,
#         campos=viewpoint_camera.camera_center,
#         prefiltered=False,
#         debug=pipe.debug
#     )

#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

#     means3D = pc.get_xyz
#     means2D = screenspace_points
#     opacity = pc.get_opacity
#     if filtered_mask is not None:
#         new_opacity = opacity.detach().clone()
#         new_opacity[filtered_mask, :] = 0
#         opacity = new_opacity

#     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
#     # scaling / rotation by the rasterizer.
#     scales = None
#     rotations = None
#     cov3D_precomp = None
#     if pipe.compute_cov3D_python:
#         cov3D_precomp = pc.get_covariance(scaling_modifier)
#     else:
#         scales = pc.get_scaling
#         rotations = pc.get_rotation

#     # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
#     # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
#     shs = None
#     colors_precomp = None
#     if override_color is None:
#         if pipe.convert_SHs_python:
#             shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
#             dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
#             dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
#             sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
#             colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
#         else:
#             shs = pc.get_features
#     else:
#         colors_precomp = override_color

#     # Rasterize visible Gaussians to image, obtain their radii (on screen). 
#     rendered_image, radii = rasterizer(
#         means3D = means3D,
#         means2D = means2D,
#         shs = shs,
#         colors_precomp = colors_precomp,
#         opacities = opacity,
#         scales = scales,
#         rotations = rotations,
#         cov3D_precomp = cov3D_precomp)

#     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#     # They will be excluded from value updates used in the splitting criteria.
#     return {"render": rendered_image,
#             "viewspace_points": screenspace_points,
#             "visibility_filter" : radii > 0,
#             "radii": radii}

def render_mask(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, precomputed_mask = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # start_time  = time.time()
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = DepthGaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    # print("Render time checker: raster_settings", time.time() - start_time)
    # start_time  = time.time()


    rasterizer = DepthGaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    mask = pc.get_mask if precomputed_mask is None else precomputed_mask

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # print("Render time checker: prepare vars", time.time() - start_time)


    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_mask, radii = rasterizer.forward_mask(
        means3D = means3D,
        means2D = means2D,
        opacities = opacity,
        mask = mask,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    # print("Render time checker: main render", time.time() - start_time)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"mask": rendered_mask,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii}


def flashsplat_render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
                      override_color=None, gt_mask=None, used_mask=None, unique_label=None, setpdb=False,
                      obj_num=2, ):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # if unique_label is not None:
    #     pdb.set_trace()

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = FlashSplat_GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        mask_grad=False,
        num_obj=obj_num,
    )

    rasterizer = FlashSplat_GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    if used_mask is not None:
        means3D = means3D[used_mask]
        opacity = opacity[used_mask]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        if used_mask is not None:
            scales = scales[used_mask]
            rotations = rotations[used_mask]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # pdb.set_trace()
            shs = pc.get_features
            if used_mask is not None:
                shs = shs[used_mask]
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if setpdb:
        pdb.set_trace()
    rendered_image, radii, depth, alpha, contrib_num, used_count, proj_xy, gs_depth = rasterizer(
        gt_mask=gt_mask,
        unique_label=unique_label,
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "alpha": alpha,
            "depth": depth,
            "contrib_num": contrib_num,
            "used_count": used_count,
            "proj_xy": proj_xy,
            "gs_depth": gs_depth}

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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._mask = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.mask_optimizer = None
        self.mask_grad_accum = torch.empty(0)
        self.mask_sign_accum = torch.empty(0)
        self.mask_val_accum = torch.empty(0)
        self.tmp_radii = None
        self._mask_lr = 0.0
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._mask,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz,
        self._mask,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_mask(self):
        return self._mask
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        mask = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self._mask = nn.Parameter(mask.requires_grad_(True))

    def reset_mask(self):
        self._mask.data = torch.zeros_like(self._mask.data)

    def mask_to_bitmap(self):
        tmp_mask = self._mask.detach().clone()
        tmp_mask[tmp_mask <= 0] = 0
        tmp_mask[tmp_mask > tmp_mask.mean()] = 1
        self._mask.data = tmp_mask

    def add_training_state(self, mask_training: bool = True):
        """Toggle which parameter set participates in the next optimization step."""
        self._mask.requires_grad_(mask_training)
        train_geometry = not mask_training
        self._xyz.requires_grad_(train_geometry)
        self._features_dc.requires_grad_(train_geometry)
        self._features_rest.requires_grad_(train_geometry)
        self._opacity.requires_grad_(train_geometry)
        self._scaling.requires_grad_(train_geometry)
        self._rotation.requires_grad_(train_geometry)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.mask_grad_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._mask_lr = training_args.mask_lr

        if self._mask.shape[0] != self._xyz.shape[0]:
            mask_tensor = torch.zeros((self._xyz.shape[0], 1), dtype=torch.float32, device=self._xyz.device)
            self._mask = nn.Parameter(mask_tensor.requires_grad_(True))
        elif not isinstance(self._mask, nn.Parameter):
            self._mask = nn.Parameter(self._mask.detach().clone().requires_grad_(True))
        elif not self._mask.requires_grad:
            self._mask.requires_grad_(True)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self._init_mask_optimizer()
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.mask_sign_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.mask_val_accum = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self._xyz.device)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'mask', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        mask = self._mask.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, mask, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_colored_ply(self, path):
        mkdir_p(os.path.dirname(path))

        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        def SH2RGB(sh):
            C0 = 0.28209479177387814
            return sh * C0 + 0.5
        
        colors = SH2RGB(f_dc)

        # N, 3
        colors = torch.clamp(torch.from_numpy(colors).squeeze().cuda(), 0.0, 1.0) * 255.
        colors = colors.detach().cpu().numpy().astype(np.uint8)

        dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        points = self._xyz.detach().cpu().numpy()
        points = [(points[i,0], points[i,1], points[i,2], colors[i,0], colors[i,1], colors[i,2]) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=dtype_full)

        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el]).write(path)


    def save_mask(self, path):
        mkdir_p(os.path.dirname(path))
        mask = self._mask.detach().cpu().numpy()        
        np.save(path, mask)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        mask = np.asarray(plydata.elements[0]["mask"])[..., np.newaxis]

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))

        self._mask = nn.Parameter(torch.tensor(mask, dtype=torch.float, device="cuda").requires_grad_(True))

        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_mask_optimizer(self, mask):
        if self.mask_optimizer is None:
            return None

        optimizable_tensors = {}
        for group in self.mask_optimizer.param_groups:
            stored_state = self.mask_optimizer.state.get(group['params'][0], None)
            current_param = group["params"][0]
            new_tensor = current_param[mask].detach()
            new_param = nn.Parameter(new_tensor.requires_grad_(True))

            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.mask_optimizer.state[group['params'][0]]
                group["params"][0] = new_param
                self.mask_optimizer.state[group['params'][0]] = stored_state
            else:
                group["params"][0] = new_param

            optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.mask_grad_accum.numel() != 0:
            self.mask_grad_accum = self.mask_grad_accum[valid_points_mask]
        if self.mask_sign_accum.numel() != 0:
            self.mask_sign_accum = self.mask_sign_accum[valid_points_mask]
        if self.mask_val_accum.numel() != 0:
            self.mask_val_accum = self.mask_val_accum[valid_points_mask]

        mask_optimizable = self._prune_mask_optimizer(valid_points_mask)
        if mask_optimizable is not None:
            self._mask = mask_optimizable["mask"]
        else:
            if self._mask.numel() != 0:
                pruned_mask = self._mask[valid_points_mask].detach().clone()
                self._mask = nn.Parameter(pruned_mask.requires_grad_(True))
            else:
                self._mask = nn.Parameter(self._mask.detach().clone().requires_grad_(True))
            self.mask_optimizer = None
            self._init_mask_optimizer()

        if self.mask_sign_accum.numel() == 0:
            self.mask_sign_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        if self.mask_val_accum.numel() == 0:
            self.mask_val_accum = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")

    def segment(self, mask=None):
        if mask is None:
            mask = self._mask > 0
        mask = mask.squeeze()
        if torch.count_nonzero(mask) == 0:
            mask = ~mask
            print("Seems like the mask is empty, segmenting the whole point cloud. Please run seg.py first.")

        if self.optimizer is None:
            self._xyz = nn.Parameter(self._xyz[mask].detach().requires_grad_(True))
            self._mask = nn.Parameter(self._mask[mask].detach().clone().requires_grad_(True))

            self._features_dc = nn.Parameter(self._features_dc[mask].detach().requires_grad_(True))
            self._features_rest = nn.Parameter(self._features_rest[mask].detach().requires_grad_(True))
            self._opacity = nn.Parameter(self._opacity[mask].detach().requires_grad_(True))
            self._scaling = nn.Parameter(self._scaling[mask].detach().requires_grad_(True))
            self._rotation = nn.Parameter(self._rotation[mask].detach().requires_grad_(True))

            self.mask_optimizer = None
            self._init_mask_optimizer()
        else:
            optimizable_tensors = self._prune_optimizer(mask)

            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

            self.xyz_gradient_accum = self.xyz_gradient_accum[mask]

            self.denom = self.denom[mask]
            if self.mask_grad_accum.numel() != 0:
                self.mask_grad_accum = self.mask_grad_accum[mask]

            mask_optimizable = self._prune_mask_optimizer(mask)
            if mask_optimizable is not None:
                self._mask = mask_optimizable["mask"]
            else:
                if self._mask.numel() != 0:
                    new_mask_tensor = self._mask[mask].detach().clone()
                else:
                    new_mask_tensor = self._mask.detach().clone()
                self._mask = nn.Parameter(new_mask_tensor.requires_grad_(True))
                self.mask_optimizer = None
                self._init_mask_optimizer()

        if self.mask_sign_accum.numel() != 0:
            self.mask_sign_accum = self.mask_sign_accum[mask]
        if self.mask_val_accum.numel() != 0:
            self.mask_val_accum = self.mask_val_accum[mask]

    def mask_prune_points(self, mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        mask_optimizable = self._prune_mask_optimizer(valid_points_mask)
        if mask_optimizable is not None and "mask" in mask_optimizable:
            self._mask = mask_optimizable["mask"]
        else:
            if self._mask.numel() != 0:
                self._mask = nn.Parameter(self._mask[valid_points_mask].detach().clone().requires_grad_(True))

        if self.xyz_gradient_accum.numel() != 0:
            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        if self.mask_grad_accum.numel() != 0:
            self.mask_grad_accum = self.mask_grad_accum[valid_points_mask]
        if self.mask_sign_accum.numel() != 0:
            self.mask_sign_accum = self.mask_sign_accum[valid_points_mask]
        if self.mask_val_accum.numel() != 0:
            self.mask_val_accum = self.mask_val_accum[valid_points_mask]

        if self.denom.numel() != 0:
            self.denom = self.denom[valid_points_mask]
        if self.max_radii2D.numel() != 0:
            self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def cat_tensors_to_mask_optimizer(self, tensors_dict):
        extension = tensors_dict.get("mask")
        if extension is None or extension.numel() == 0:
            return {"mask": self._mask}

        if self.mask_optimizer is None:
            combined = torch.cat((self._mask.detach().clone(), extension.detach().clone()), dim=0)
            self._mask = nn.Parameter(combined.requires_grad_(True))
            self._init_mask_optimizer()
            return {"mask": self._mask}

        optimizable_tensors = {}
        for group in self.mask_optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.mask_optimizer.state.get(group['params'][0], None)
            current_param = group["params"][0]

            combined_tensor = torch.cat((current_param.detach(), extension_tensor.detach()), dim=0)
            new_param = nn.Parameter(combined_tensor.requires_grad_(True))

            if stored_state is not None:
                zeros_ext = torch.zeros_like(extension_tensor)
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], zeros_ext), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], zeros_ext), dim=0)

                del self.mask_optimizer.state[group['params'][0]]
                group["params"][0] = new_param
                self.mask_optimizer.state[group['params'][0]] = stored_state
            else:
                group["params"][0] = new_param

            optimizable_tensors[group["name"]] = group["params"][0]

        if "mask" in optimizable_tensors:
            self._mask = optimizable_tensors["mask"]
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_mask, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        mask_optimizable = self.cat_tensors_to_mask_optimizer({"mask": new_mask})
        if mask_optimizable is not None and "mask" in mask_optimizable:
            self._mask = mask_optimizable["mask"]

        device = self._xyz.device
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)
        self.mask_grad_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.mask_sign_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.mask_val_accum = torch.zeros((self.get_xyz.shape[0], 2), device=device)

    def mask_densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_mask):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        d_mask = {"mask": new_mask}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        mask_optimizable = self.cat_tensors_to_mask_optimizer(d_mask)
        if mask_optimizable is not None and "mask" in mask_optimizable:
            self._mask = mask_optimizable["mask"]

        device = self._xyz.device
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.mask_sign_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.mask_val_accum = torch.zeros((self.get_xyz.shape[0], 2), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)
        self.mask_grad_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

        new_mask = self._mask[selected_pts_mask].repeat(N,1)

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_mask, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]

        new_mask = self._mask[selected_pts_mask]

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_mask, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def add_mask_signal_densification_stats(self, mask_signals):
        if mask_signals is None:
            return

        if hasattr(mask_signals, "grad") and mask_signals.grad is not None:
            mask_signal_grads = mask_signals.grad.detach()
        else:
            mask_signal_grads = mask_signals.detach()

        if mask_signal_grads.numel() == 0:
            return

        if mask_signal_grads.dim() == 1 or mask_signal_grads.shape[1] != 2:
            mask_signal_grads = mask_signal_grads.reshape(-1, 1)
            mask_signal_grads = torch.cat((mask_signal_grads, torch.zeros_like(mask_signal_grads)), dim=1)

        relative_difference = (mask_signal_grads[:, 0] - mask_signal_grads[:, 1]) / (
                mask_signal_grads[:, 0] + mask_signal_grads[:, 1] + 1e-6)
        if self.mask_val_accum.shape[0] != mask_signal_grads.shape[0]:
            self.mask_val_accum = torch.zeros((mask_signal_grads.shape[0], 2), device=mask_signal_grads.device)
        if self.mask_sign_accum.shape[0] != mask_signal_grads.shape[0]:
            self.mask_sign_accum = torch.zeros((mask_signal_grads.shape[0], 1), device=mask_signal_grads.device)

        self.mask_val_accum[:, 0] += mask_signal_grads[:, 0]
        self.mask_val_accum[:, 1] += mask_signal_grads[:, 1]

        self.mask_sign_accum[:] += relative_difference.abs().unsqueeze(1)
        zero_mask = (mask_signal_grads[:, 0] < 1e-6) & (mask_signal_grads[:, 1] < 1e-6)
        self.mask_sign_accum[zero_mask] += 1.0

    def _init_mask_optimizer(self):
        if not isinstance(self._mask, nn.Parameter):
            self._mask = nn.Parameter(self._mask.detach().clone().requires_grad_(True))
        elif not self._mask.requires_grad:
            self._mask.requires_grad_(True)

        if self._mask_lr <= 0:
            self._mask_lr = 1.0

        mask_group = [{'params': [self._mask], 'lr': self._mask_lr, "name": "mask"}]

        if self.mask_optimizer is None:
            self.mask_optimizer = torch.optim.Adam(mask_group, lr=0.0, eps=1e-15)
        else:
            for group in self.mask_optimizer.param_groups:
                group['params'][0] = self._mask
                group['lr'] = self._mask_lr

    def reset_mask_statistics(self):
        if self._mask.numel() == 0:
            return
        self.mask_grad_accum = torch.zeros((self._mask.shape[0], 1), device=self._mask.device)

    def accumulate_mask_gradients(self):
        if self._mask.grad is None:
            return
        if self.mask_grad_accum.numel() == 0 or self.mask_grad_accum.shape[0] != self._mask.shape[0]:
            self.mask_grad_accum = torch.zeros((self._mask.shape[0], 1), device=self._mask.device)
        self.mask_grad_accum += self._mask.grad.detach().abs()

    def mask_and_split(self, mask_signals_threshold, scene_extent, base_num=None, prune_only=False, split_factor=2):
        stats = {"selected": 0, "split": 0, "pruned": 0}

        if self.mask_sign_accum.numel() == 0 or self.mask_val_accum.numel() == 0:
            return stats

        if base_num is None or base_num <= 0:
            base_num = max(int(split_factor), 1)

        mean_relative_difference = self.mask_sign_accum / max(base_num, 1)
        distance_mask = (mean_relative_difference < mask_signals_threshold)

        selected_pts_mask = distance_mask.squeeze()
        stats["selected"] = int(selected_pts_mask.sum().item())
        if prune_only:
            if selected_pts_mask.any():
                self.mask_prune_points(selected_pts_mask)
                stats["pruned"] = stats["selected"]
            self.mask_sign_accum = torch.zeros_like(self.mask_sign_accum)
            self.mask_val_accum = torch.zeros_like(self.mask_val_accum)
            return stats

        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )

        if not torch.any(selected_pts_mask):
            self.mask_sign_accum = torch.zeros_like(self.mask_sign_accum)
            self.mask_val_accum = torch.zeros_like(self.mask_val_accum)
            return stats

        device = self._xyz.device
        stds = self.get_scaling[selected_pts_mask].repeat(split_factor, 1)
        means = torch.zeros((stds.size(0), 3), device=device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(split_factor, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(split_factor, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(split_factor, 1) / (0.8 * split_factor))
        new_rotation = self._rotation[selected_pts_mask].repeat(split_factor, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(split_factor, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(split_factor, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(split_factor, 1)
        new_mask = torch.zeros_like(self._mask[selected_pts_mask]).repeat(split_factor, 1)

        self.mask_densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_mask)

        selected_count = int(selected_pts_mask.sum().item())
        stats["pruned"] = selected_count
        stats["split"] = int(selected_count * split_factor)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(split_factor * selected_pts_mask.sum(), device=device, dtype=bool)))
        self.mask_prune_points(prune_filter)
        self.mask_sign_accum = torch.zeros_like(self.mask_sign_accum)
        self.mask_val_accum = torch.zeros_like(self.mask_val_accum)
        return stats
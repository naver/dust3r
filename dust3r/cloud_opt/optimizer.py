# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Main class for the implementation of the global alignment
# --------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn

from dust3r.cloud_opt.base_opt import BasePCOptimizer
from dust3r.utils.geometry import xy_grid, geotrf
from dust3r.utils.device import to_cpu, to_numpy


class PointCloudOptimizer(BasePCOptimizer):
    """ Optimize a global scene, given a list of pairwise observations.
    Graph node: images
    Graph edges: observations = (pred1, pred2)
    """

    def __init__(self, *args, optimize_pp=False, focal_break=20, **kwargs):
        super().__init__(*args, **kwargs)

        self.has_im_poses = True  # by definition of this class
        self.focal_break = focal_break

        # adding thing to optimize
        self.im_depthmaps = nn.ParameterList(torch.randn(H, W)/10-3 for H, W in self.imshapes)  # log(depth)
        self.im_poses = nn.ParameterList(self.rand_pose(self.POSE_DIM) for _ in range(self.n_imgs))  # camera poses
        self.im_focals = nn.ParameterList(torch.FloatTensor(
            [self.focal_break*np.log(max(H, W))]) for H, W in self.imshapes)  # camera intrinsics
        self.im_pp = nn.ParameterList(torch.zeros((2,)) for _ in range(self.n_imgs))  # camera intrinsics
        self.im_pp.requires_grad_(optimize_pp)

        self.imshape = self.imshapes[0]
        im_areas = [h*w for h, w in self.imshapes]
        self.max_area = max(im_areas)

        # adding thing to optimize
        self.im_depthmaps = ParameterStack(self.im_depthmaps, is_param=True, fill=self.max_area)
        self.im_poses = ParameterStack(self.im_poses, is_param=True)
        self.im_focals = ParameterStack(self.im_focals, is_param=True)
        self.im_pp = ParameterStack(self.im_pp, is_param=True)
        self.register_buffer('_pp', torch.tensor([(w/2, h/2) for h, w in self.imshapes]))
        self.register_buffer('_grid', ParameterStack(
            [xy_grid(W, H, device=self.device) for H, W in self.imshapes], fill=self.max_area))

        # pre-compute pixel weights
        self.register_buffer('_weight_i', ParameterStack(
            [self.conf_trf(self.conf_i[i_j]) for i_j in self.str_edges], fill=self.max_area))
        self.register_buffer('_weight_j', ParameterStack(
            [self.conf_trf(self.conf_j[i_j]) for i_j in self.str_edges], fill=self.max_area))

        # precompute aa
        self.register_buffer('_stacked_pred_i', ParameterStack(self.pred_i, self.str_edges, fill=self.max_area))
        self.register_buffer('_stacked_pred_j', ParameterStack(self.pred_j, self.str_edges, fill=self.max_area))
        self.register_buffer('_ei', torch.tensor([i for i, j in self.edges]))
        self.register_buffer('_ej', torch.tensor([j for i, j in self.edges]))
        self.total_area_i = sum([im_areas[i] for i, j in self.edges])
        self.total_area_j = sum([im_areas[j] for i, j in self.edges])

    def _check_all_imgs_are_selected(self, msk):
        assert np.all(self._get_msk_indices(msk) == np.arange(self.n_imgs)), 'incomplete mask!'

    def preset_pose(self, known_poses, pose_msk=None):  # cam-to-world
        self._check_all_imgs_are_selected(pose_msk)

        if isinstance(known_poses, torch.Tensor) and known_poses.ndim == 2:
            known_poses = [known_poses]
        for idx, pose in zip(self._get_msk_indices(pose_msk), known_poses):
            print(f' (setting pose #{idx} = {pose[:3,3]})')
            self._no_grad(self._set_pose(self.im_poses, idx, torch.tensor(pose)))

        # normalize scale if there's less than 1 known pose
        n_known_poses = sum((p.requires_grad is False) for p in self.im_poses)
        self.norm_pw_scale = (n_known_poses <= 1)

        self.im_poses.requires_grad_(False)
        self.norm_pw_scale = False

    def preset_focal(self, known_focals, msk=None):
        self._check_all_imgs_are_selected(msk)

        for idx, focal in zip(self._get_msk_indices(msk), known_focals):
            print(f' (setting focal #{idx} = {focal})')
            self._no_grad(self._set_focal(idx, focal))

        self.im_focals.requires_grad_(False)

    def preset_principal_point(self, known_pp, msk=None):
        self._check_all_imgs_are_selected(msk)

        for idx, pp in zip(self._get_msk_indices(msk), known_pp):
            print(f' (setting principal point #{idx} = {pp})')
            self._no_grad(self._set_principal_point(idx, pp))

        self.im_pp.requires_grad_(False)

    def _no_grad(self, tensor):
        assert tensor.requires_grad, 'it must be True at this point, otherwise no modification occurs'

    def _set_focal(self, idx, focal, force=False):
        param = self.im_focals[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = self.focal_break * np.log(focal)
        return param

    def get_focals(self):
        log_focals = torch.stack(list(self.im_focals), dim=0)
        return (log_focals / self.focal_break).exp()

    def get_known_focal_mask(self):
        return torch.tensor([not (p.requires_grad) for p in self.im_focals])

    def _set_principal_point(self, idx, pp, force=False):
        param = self.im_pp[idx]
        H, W = self.imshapes[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = to_cpu(to_numpy(pp) - (W/2, H/2)) / 10
        return param

    def get_principal_points(self):
        return self._pp + 10 * self.im_pp

    def get_intrinsics(self):
        K = torch.zeros((self.n_imgs, 3, 3), device=self.device)
        focals = self.get_focals().flatten()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, :2, 2] = self.get_principal_points()
        K[:, 2, 2] = 1
        return K

    def get_im_poses(self):  # cam to world
        cam2world = self._get_poses(self.im_poses)
        return cam2world

    def _set_depthmap(self, idx, depth, force=False):
        depth = _ravel_hw(depth, self.max_area)

        param = self.im_depthmaps[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = depth.log().nan_to_num(neginf=0)
        return param

    def get_depthmaps(self, raw=False):
        res = self.im_depthmaps.exp()
        if not raw:
            res = [dm[:h*w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def depth_to_pts3d(self):
        # Get depths and  projection params if not provided
        focals = self.get_focals()
        pp = self.get_principal_points()
        im_poses = self.get_im_poses()
        depth = self.get_depthmaps(raw=True)

        # get pointmaps in camera frame
        rel_ptmaps = _fast_depthmap_to_pts3d(depth, self._grid, focals, pp=pp)
        # project to world frame
        return geotrf(im_poses, rel_ptmaps)

    def get_pts3d(self, raw=False):
        res = self.depth_to_pts3d()
        if not raw:
            res = [dm[:h*w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def forward(self):
        pw_poses = self.get_pw_poses()  # cam-to-world
        pw_adapt = self.get_adaptors().unsqueeze(1)
        proj_pts3d = self.get_pts3d(raw=True)

        # rotate pairwise prediction according to pw_poses
        aligned_pred_i = geotrf(pw_poses, pw_adapt * self._stacked_pred_i)
        aligned_pred_j = geotrf(pw_poses, pw_adapt * self._stacked_pred_j)

        # compute the less
        li = self.dist(proj_pts3d[self._ei], aligned_pred_i, weight=self._weight_i).sum() / self.total_area_i
        lj = self.dist(proj_pts3d[self._ej], aligned_pred_j, weight=self._weight_j).sum() / self.total_area_j

        return li + lj


def _fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1)
    assert focal.shape == (len(depth), 1, 1)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == depth.shape + (2,)
    depth = depth.unsqueeze(-1)
    return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)


def ParameterStack(params, keys=None, is_param=None, fill=0):
    if keys is not None:
        params = [params[k] for k in keys]

    if fill > 0:
        params = [_ravel_hw(p, fill) for p in params]

    requires_grad = params[0].requires_grad
    assert all(p.requires_grad == requires_grad for p in params)

    params = torch.stack(list(params)).float().detach()
    if is_param or requires_grad:
        params = nn.Parameter(params)
        params.requires_grad_(requires_grad)
    return params


def _ravel_hw(tensor, fill=0):
    # ravel H,W
    tensor = tensor.view((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])

    if len(tensor) < fill:
        tensor = torch.cat((tensor, tensor.new_zeros((fill - len(tensor),)+tensor.shape[1:])))
    return tensor


def acceptable_focal_range(H, W, minf=0.5, maxf=3.5):
    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    return minf*focal_base, maxf*focal_base


def apply_mask(img, msk):
    img = img.copy()
    img[msk] = 0
    return img

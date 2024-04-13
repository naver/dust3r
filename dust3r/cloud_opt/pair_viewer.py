# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dummy optimizer for visualizing pairs
# --------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import cv2

from dust3r.cloud_opt.base_opt import BasePCOptimizer
from dust3r.utils.geometry import inv, geotrf, depthmap_to_absolute_camera_coordinates
from dust3r.cloud_opt.commons import edge_str
from dust3r.post_process import estimate_focal_knowing_depth


class PairViewer (BasePCOptimizer):
    """
    This a Dummy Optimizer.
    To use only when the goal is to visualize the results for a pair of images (with is_symmetrized)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.is_symmetrized and self.n_edges == 2
        self.has_im_poses = True

        # compute all parameters directly from raw input
        self.focals = []
        self.pp = []
        rel_poses = []
        confs = []
        for i in range(self.n_imgs):
            conf = float(self.conf_i[edge_str(i, 1-i)].mean() * self.conf_j[edge_str(i, 1-i)].mean())
            if self.verbose:
                print(f'  - {conf=:.3} for edge {i}-{1-i}')
            confs.append(conf)

            H, W = self.imshapes[i]
            pts3d = self.pred_i[edge_str(i, 1-i)]
            pp = torch.tensor((W/2, H/2))
            focal = float(estimate_focal_knowing_depth(pts3d[None], pp, focal_mode='weiszfeld'))
            self.focals.append(focal)
            self.pp.append(pp)

            # estimate the pose of pts1 in image 2
            pixels = np.mgrid[:W, :H].T.astype(np.float32)
            pts3d = self.pred_j[edge_str(1-i, i)].numpy()
            assert pts3d.shape[:2] == (H, W)
            msk = self.get_masks()[i].numpy()
            K = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])

            try:
                res = cv2.solvePnPRansac(pts3d[msk], pixels[msk], K, None,
                                         iterationsCount=100, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
                success, R, T, inliers = res
                assert success

                R = cv2.Rodrigues(R)[0]  # world to cam
                pose = inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])  # cam to world
            except:
                pose = np.eye(4)
            rel_poses.append(torch.from_numpy(pose.astype(np.float32)))

        # let's use the pair with the most confidence
        if confs[0] > confs[1]:
            # ptcloud is expressed in camera1
            self.im_poses = [torch.eye(4), rel_poses[1]]  # I, cam2-to-cam1
            self.depth = [self.pred_i['0_1'][..., 2], geotrf(inv(rel_poses[1]), self.pred_j['0_1'])[..., 2]]
        else:
            # ptcloud is expressed in camera2
            self.im_poses = [rel_poses[0], torch.eye(4)]  # I, cam1-to-cam2
            self.depth = [geotrf(inv(rel_poses[0]), self.pred_j['1_0'])[..., 2], self.pred_i['1_0'][..., 2]]

        self.im_poses = nn.Parameter(torch.stack(self.im_poses, dim=0), requires_grad=False)
        self.focals = nn.Parameter(torch.tensor(self.focals), requires_grad=False)
        self.pp = nn.Parameter(torch.stack(self.pp, dim=0), requires_grad=False)
        self.depth = nn.ParameterList(self.depth)
        for p in self.parameters():
            p.requires_grad = False

    def _set_depthmap(self, idx, depth, force=False):
        if self.verbose:
            print('_set_depthmap is ignored in PairViewer')
        return

    def get_depthmaps(self, raw=False):
        depth = [d.to(self.device) for d in self.depth]
        return depth

    def _set_focal(self, idx, focal, force=False):
        self.focals[idx] = focal

    def get_focals(self):
        return self.focals

    def get_known_focal_mask(self):
        return torch.tensor([not (p.requires_grad) for p in self.focals])

    def get_principal_points(self):
        return self.pp

    def get_intrinsics(self):
        focals = self.get_focals()
        pps = self.get_principal_points()
        K = torch.zeros((len(focals), 3, 3), device=self.device)
        for i in range(len(focals)):
            K[i, 0, 0] = K[i, 1, 1] = focals[i]
            K[i, :2, 2] = pps[i]
            K[i, 2, 2] = 1
        return K

    def get_im_poses(self):
        return self.im_poses

    def depth_to_pts3d(self):
        pts3d = []
        for d, intrinsics, im_pose in zip(self.depth, self.get_intrinsics(), self.get_im_poses()):
            pts, _ = depthmap_to_absolute_camera_coordinates(d.cpu().numpy(),
                                                             intrinsics.cpu().numpy(),
                                                             im_pose.cpu().numpy())
            pts3d.append(torch.from_numpy(pts).to(device=self.device))
        return pts3d

    def forward(self):
        return float('nan')

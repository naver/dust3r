# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Implementation of DUSt3R training losses
# --------------------------------------------------------
import collections
from copy import copy, deepcopy

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn

from apple_model.apple_optimizer import AppleOptimizer
from dust3r.inference import get_pred_pts3d
from dust3r.utils.geometry import get_joint_pointcloud_depth, get_joint_pointcloud_center_scale
from dust3r.utils.geometry import inv, geotrf, normalize_pointcloud
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import collate_with_cat


def Sum(*losses_and_masks):
    loss, mask = losses_and_masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        return losses_and_masks
    else:
        # we are returning the global loss
        for loss2, mask2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss


class BaseCriterion(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction


class LLoss(BaseCriterion):
    """ L-norm loss
    """

    def forward(self, a, b):
        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3, f'Bad shape = {a.shape}'
        dist = self.distance(a, b)
        assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f'bad {self.reduction=} mode')

    def distance(self, a, b):
        raise NotImplementedError()


class L21Loss(LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance


L21 = L21Loss(reduction='none')


class Criterion(nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(criterion, BaseCriterion), f'{criterion} is not a proper criterion!'
        self.criterion = copy(criterion)

    def get_name(self):
        return f'{type(self).__name__}({self.criterion})'

    def with_reduction(self, mode='none'):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = mode  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss(nn.Module):
    """ Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res

    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f'{self._alpha:g}*{name}'
        if self._loss2:
            name = f'{name} + {self._loss2}'
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details


class Regr3D(Criterion, MultiLoss):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, dist_clip=None):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gt1['camera_pose'])
        gt_pts1 = geotrf(in_camera1, gt1['pts3d'])  # B,H,W,3
        gt_pts2 = geotrf(in_camera1, gt2['pts3d'])  # B,H,W,3

        valid1 = gt1['valid_mask'].clone()
        valid2 = gt2['valid_mask'].clone()

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
            dis2 = gt_pts2.norm(dim=-1)  # (B, H, W)
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2 = valid2 & (dis2 <= dist_clip)

        pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False)
        pr_pts2 = get_pred_pts3d(gt2, pred2, use_pose=True)

        # normalize 3d points
        if self.norm_mode:
            pr_pts1, pr_pts2 = normalize_pointcloud(pr_pts1, pr_pts2, self.norm_mode, valid1, valid2)
        if self.norm_mode and not self.gt_scale:
            gt_pts1, gt_pts2 = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode, valid1, valid2)

        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, {}

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = \
            self.get_all_pts3d(gt1, gt2, pred1, pred2, **kw)
        # loss on img1 side
        l1 = self.criterion(pred_pts1[mask1], gt_pts1[mask1])
        # loss on gt2 side
        l2 = self.criterion(pred_pts2[mask2], gt_pts2[mask2])
        self_name = type(self).__name__
        details = {self_name + '_pts3d_1': float(l1.mean()), self_name + '_pts3d_2': float(l2.mean())}
        return Sum((l1, mask1), (l2, mask2)), (details | monitoring)


class CorrespondenceLoss(Criterion, MultiLoss):
    def __init__(self, criterion, fitness_threshold=0.25, use_conf=True):
        super().__init__(criterion)
        self.loos_func = nn.MSELoss()
        self.fitness_threshold = fitness_threshold
        self.use_conf = use_conf

    @staticmethod
    def get_conf_log(x):
        return x, torch.log(x)

    @staticmethod
    def make_copy_and_auto_grad(item):
        item_copy = deepcopy(item)
        keyword = ['pts3d', 'conf', 'img']
        for k, v in item_copy.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                item_copy[k] = v.requires_grad_(True)
        return item_copy

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        # global alignment
        gt1_copy = self.make_copy_and_auto_grad(gt1)
        gt2_copy = self.make_copy_and_auto_grad(gt2)
        gt1_copy['idx'] = [0, 1]
        gt2_copy['idx'] = [1, 0]

        output = {'view1': gt1_copy, 'view2': gt2_copy, 'pred1': pred1, 'pred2': pred2}
        scene = global_aligner(output, device='cuda', mode=GlobalAlignerMode.PointCloudOptimizer)
        _loss = scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.01)

        # retrieve useful values from scene:
        pts3d = scene.get_pts3d()
        masks = gt1['mask'].detach().cpu().tolist()
        conf_masks_, conf_mask_log = self.get_conf_log(pred1['conf'])
        conf_masks = [conf_masks_[0], conf_masks_[1]]

        points_with_grad = []
        conf = []
        for pts, mask, conf_mask in zip(pts3d, masks, conf_masks):
            for i in range(pts.shape[0]):
                for j in range(pts.shape[1]):
                    if mask[i][j] == 255:
                        points_with_grad.append(pts[i][j])
                        conf.append(conf_mask[i][j])

        points = np.asarray(list(map(lambda x: x.detach().cpu(), points_with_grad)))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        optimizer = AppleOptimizer(target=pcd)
        optimizer.opt_with_cpd()
        reg_res = optimizer.reg_res
        loss = torch.zeros(1)
        if optimizer.reg_res.fitness < self.fitness_threshold:
            return loss
        correspondence = np.asarray(reg_res.correspondence_set)
        matching_dict = collections.defaultdict(list)
        inp = []
        target = []
        conf_values = []
        # source from ideal model, target from prediction
        for source_idx, target_idx in correspondence:
            matching_dict[target_idx].append(optimizer.optimized_model.pcd.points[source_idx])
        for target_idx, p_list in matching_dict.items():
            inp.append(points_with_grad[target_idx])
            conf_values.append(conf[target_idx])
            target.append(np.mean(np.asarray(p_list), axis=0))
        conf_values = torch.stack(conf_values)
        loss = self.criterion(torch.stack(inp), torch.tensor(np.asarray(target)).cuda())

        if self.use_conf:
            loss = torch.matmul(loss.float(), conf_values) / loss.shape[0]
        else:
            loss = loss.mean()

        self_name = type(self).__name__
        details = {self_name + '_apple_loss': float(loss.mean())}
        return loss, details


class ConfLoss(MultiLoss):
    """ Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10) 

        alpha: hyperparameter
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction('none')
        # self.apple_align_loss =

    def get_name(self):
        return f'ConfLoss({self.pixel_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        # compute per-pixel loss
        ((loss1, msk1), (loss2, msk2)), details = self.pixel_loss(gt1, gt2, pred1, pred2, **kw)
        if loss1.numel() == 0:
            print('NO VALID POINTS in img1', force=True)
        if loss2.numel() == 0:
            print('NO VALID POINTS in img2', force=True)

        # weight by confidence
        conf1, log_conf1 = self.get_conf_log(pred1['conf'][msk1])
        conf2, log_conf2 = self.get_conf_log(pred2['conf'][msk2])
        conf_loss1 = loss1 * conf1 - self.alpha * log_conf1
        conf_loss2 = loss2 * conf2 - self.alpha * log_conf2

        # average + nan protection (in case of no valid pixels at all)
        conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0
        conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0

        return conf_loss1 + conf_loss2, dict(conf_loss_1=float(conf_loss1), conf_loss2=float(conf_loss2), **details)


class Regr3D_ShiftInv(Regr3D):
    """ Same than Regr3D but invariant to depth shift.
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute unnormalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = \
            super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # compute median depth
        gt_z1, gt_z2 = gt_pts1[..., 2], gt_pts2[..., 2]
        pred_z1, pred_z2 = pred_pts1[..., 2], pred_pts2[..., 2]
        gt_shift_z = get_joint_pointcloud_depth(gt_z1, gt_z2, mask1, mask2)[:, None, None]
        pred_shift_z = get_joint_pointcloud_depth(pred_z1, pred_z2, mask1, mask2)[:, None, None]

        # subtract the median depth
        gt_z1 -= gt_shift_z
        gt_z2 -= gt_shift_z
        pred_z1 -= pred_shift_z
        pred_z2 -= pred_shift_z

        # monitoring = dict(monitoring, gt_shift_z=gt_shift_z.mean().detach(), pred_shift_z=pred_shift_z.mean().detach())
        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring


class Regr3D_ScaleInv(Regr3D):
    """ Same than Regr3D but invariant to depth shift.
        if gt_scale == True: enforce the prediction to take the same scale than GT
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute depth-normalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # measure scene scale
        _, gt_scale = get_joint_pointcloud_center_scale(gt_pts1, gt_pts2, mask1, mask2)
        _, pred_scale = get_joint_pointcloud_center_scale(pred_pts1, pred_pts2, mask1, mask2)

        # prevent predictions to be in a ridiculous range
        pred_scale = pred_scale.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:
            pred_pts1 *= gt_scale / pred_scale
            pred_pts2 *= gt_scale / pred_scale
            # monitoring = dict(monitoring, pred_scale=(pred_scale/gt_scale).mean())
        else:
            gt_pts1 /= gt_scale
            gt_pts2 /= gt_scale
            pred_pts1 /= pred_scale
            pred_pts2 /= pred_scale
            # monitoring = dict(monitoring, gt_scale=gt_scale.mean(), pred_scale=pred_scale.mean().detach())

        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring


class Regr3D_ScaleShiftInv(Regr3D_ScaleInv, Regr3D_ShiftInv):
    # calls Regr3D_ShiftInv first, then Regr3D_ScaleInv
    pass

# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Initialization functions for global alignment
# --------------------------------------------------------
from functools import cache

import numpy as np
import scipy.sparse as sp
import torch
import cv2
import roma
from tqdm import tqdm

from dust3r.utils.geometry import geotrf, inv, get_med_dist_between_poses
from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.viz import to_numpy

from dust3r.cloud_opt.commons import edge_str, i_j_ij, compute_edge_scores


@torch.no_grad()
def init_from_known_poses(self, niter_PnP=10, min_conf_thr=3):
    device = self.device

    # indices of known poses
    nkp, known_poses_msk, known_poses = get_known_poses(self)
    assert nkp == self.n_imgs, 'not all poses are known'

    # get all focals
    nkf, _, im_focals = get_known_focals(self)
    assert nkf == self.n_imgs
    im_pp = self.get_principal_points()

    best_depthmaps = {}
    # init all pairwise poses
    for e, (i, j) in enumerate(tqdm(self.edges, disable=not self.verbose)):
        i_j = edge_str(i, j)

        # find relative pose for this pair
        P1 = torch.eye(4, device=device)
        msk = self.conf_i[i_j] > min(min_conf_thr, self.conf_i[i_j].min() - 0.1)
        _, P2 = fast_pnp(self.pred_j[i_j], float(im_focals[i].mean()),
                         pp=im_pp[i], msk=msk, device=device, niter_PnP=niter_PnP)

        # align the two predicted camera with the two gt cameras
        s, R, T = align_multiple_poses(torch.stack((P1, P2)), known_poses[[i, j]])
        # normally we have known_poses[i] ~= sRT_to_4x4(s,R,T,device) @ P1
        # and geotrf(sRT_to_4x4(1,R,T,device), s*P2[:3,3])
        self._set_pose(self.pw_poses, e, R, T, scale=s)

        # remember if this is a good depthmap
        score = float(self.conf_i[i_j].mean())
        if score > best_depthmaps.get(i, (0,))[0]:
            best_depthmaps[i] = score, i_j, s

    # init all image poses
    for n in range(self.n_imgs):
        assert known_poses_msk[n]
        _, i_j, scale = best_depthmaps[n]
        depth = self.pred_i[i_j][:, :, 2]
        self._set_depthmap(n, depth * scale)


@torch.no_grad()
def init_minimum_spanning_tree(self, **kw):
    """ Init all camera poses (image-wise and pairwise poses) given
        an initial set of pairwise estimations.
    """
    device = self.device
    pts3d, _, im_focals, im_poses = minimum_spanning_tree(self.imshapes, self.edges,
                                                          self.pred_i, self.pred_j, self.conf_i, self.conf_j, self.im_conf, self.min_conf_thr,
                                                          device, has_im_poses=self.has_im_poses, verbose=self.verbose,
                                                          **kw)

    return init_from_pts3d(self, pts3d, im_focals, im_poses)


def init_from_pts3d(self, pts3d, im_focals, im_poses):
    # init poses
    nkp, known_poses_msk, known_poses = get_known_poses(self)
    if nkp == 1:
        raise NotImplementedError("Would be simpler to just align everything afterwards on the single known pose")
    elif nkp > 1:
        # global rigid SE3 alignment
        s, R, T = align_multiple_poses(im_poses[known_poses_msk], known_poses[known_poses_msk])
        trf = sRT_to_4x4(s, R, T, device=known_poses.device)

        # rotate everything
        im_poses = trf @ im_poses
        im_poses[:, :3, :3] /= s  # undo scaling on the rotation part
        for img_pts3d in pts3d:
            img_pts3d[:] = geotrf(trf, img_pts3d)

    # set all pairwise poses
    for e, (i, j) in enumerate(self.edges):
        i_j = edge_str(i, j)
        # compute transform that goes from cam to world
        s, R, T = rigid_points_registration(self.pred_i[i_j], pts3d[i], conf=self.conf_i[i_j])
        self._set_pose(self.pw_poses, e, R, T, scale=s)

    # take into account the scale normalization
    s_factor = self.get_pw_norm_scale_factor()
    im_poses[:, :3, 3] *= s_factor  # apply downscaling factor
    for img_pts3d in pts3d:
        img_pts3d *= s_factor

    # init all image poses
    if self.has_im_poses:
        for i in range(self.n_imgs):
            cam2world = im_poses[i]
            depth = geotrf(inv(cam2world), pts3d[i])[..., 2]
            self._set_depthmap(i, depth)
            self._set_pose(self.im_poses, i, cam2world)
            if im_focals[i] is not None:
                self._set_focal(i, im_focals[i])

    if self.verbose:
        print(' init loss =', float(self()))


def minimum_spanning_tree(imshapes, edges, pred_i, pred_j, conf_i, conf_j, im_conf, min_conf_thr,
                          device, has_im_poses=True, niter_PnP=10, verbose=True):
    n_imgs = len(imshapes)
    sparse_graph = -dict_to_sparse_graph(compute_edge_scores(map(i_j_ij, edges), conf_i, conf_j))
    msp = sp.csgraph.minimum_spanning_tree(sparse_graph).tocoo()

    # temp variable to store 3d points
    pts3d = [None] * len(imshapes)

    todo = sorted(zip(-msp.data, msp.row, msp.col))  # sorted edges
    im_poses = [None] * n_imgs
    im_focals = [None] * n_imgs

    # init with strongest edge
    score, i, j = todo.pop()
    if verbose:
        print(f' init edge ({i}*,{j}*) {score=}')
    i_j = edge_str(i, j)
    pts3d[i] = pred_i[i_j].clone()
    pts3d[j] = pred_j[i_j].clone()
    done = {i, j}
    if has_im_poses:
        im_poses[i] = torch.eye(4, device=device)
        im_focals[i] = estimate_focal(pred_i[i_j])

    # set initial pointcloud based on pairwise graph
    msp_edges = [(i, j)]
    while todo:
        # each time, predict the next one
        score, i, j = todo.pop()

        if im_focals[i] is None:
            im_focals[i] = estimate_focal(pred_i[i_j])

        if i in done:
            if verbose:
                print(f' init edge ({i},{j}*) {score=}')
            assert j not in done
            # align pred[i] with pts3d[i], and then set j accordingly
            i_j = edge_str(i, j)
            s, R, T = rigid_points_registration(pred_i[i_j], pts3d[i], conf=conf_i[i_j])
            trf = sRT_to_4x4(s, R, T, device)
            pts3d[j] = geotrf(trf, pred_j[i_j])
            done.add(j)
            msp_edges.append((i, j))

            if has_im_poses and im_poses[i] is None:
                im_poses[i] = sRT_to_4x4(1, R, T, device)

        elif j in done:
            if verbose:
                print(f' init edge ({i}*,{j}) {score=}')
            assert i not in done
            i_j = edge_str(i, j)
            s, R, T = rigid_points_registration(pred_j[i_j], pts3d[j], conf=conf_j[i_j])
            trf = sRT_to_4x4(s, R, T, device)
            pts3d[i] = geotrf(trf, pred_i[i_j])
            done.add(i)
            msp_edges.append((i, j))

            if has_im_poses and im_poses[i] is None:
                im_poses[i] = sRT_to_4x4(1, R, T, device)
        else:
            # let's try again later
            todo.insert(0, (score, i, j))

    if has_im_poses:
        # complete all missing informations
        pair_scores = list(sparse_graph.values())  # already negative scores: less is best
        edges_from_best_to_worse = np.array(list(sparse_graph.keys()))[np.argsort(pair_scores)]
        for i, j in edges_from_best_to_worse.tolist():
            if im_focals[i] is None:
                im_focals[i] = estimate_focal(pred_i[edge_str(i, j)])

        for i in range(n_imgs):
            if im_poses[i] is None:
                msk = im_conf[i] > min_conf_thr
                res = fast_pnp(pts3d[i], im_focals[i], msk=msk, device=device, niter_PnP=niter_PnP)
                if res:
                    im_focals[i], im_poses[i] = res
            if im_poses[i] is None:
                im_poses[i] = torch.eye(4, device=device)
        im_poses = torch.stack(im_poses)
    else:
        im_poses = im_focals = None

    return pts3d, msp_edges, im_focals, im_poses


def dict_to_sparse_graph(dic):
    n_imgs = max(max(e) for e in dic) + 1
    res = sp.dok_array((n_imgs, n_imgs))
    for edge, value in dic.items():
        res[edge] = value
    return res


def rigid_points_registration(pts1, pts2, conf):
    R, T, s = roma.rigid_points_registration(
        pts1.reshape(-1, 3), pts2.reshape(-1, 3), weights=conf.ravel(), compute_scaling=True)
    return s, R, T  # return un-scaled (R, T)


def sRT_to_4x4(scale, R, T, device):
    trf = torch.eye(4, device=device)
    trf[:3, :3] = R * scale
    trf[:3, 3] = T.ravel()  # doesn't need scaling
    return trf


def estimate_focal(pts3d_i, pp=None):
    if pp is None:
        H, W, THREE = pts3d_i.shape
        assert THREE == 3
        pp = torch.tensor((W/2, H/2), device=pts3d_i.device)
    focal = estimate_focal_knowing_depth(pts3d_i.unsqueeze(0), pp.unsqueeze(0), focal_mode='weiszfeld').ravel()
    return float(focal)


@cache
def pixel_grid(H, W):
    return np.mgrid[:W, :H].T.astype(np.float32)


def fast_pnp(pts3d, focal, msk, device, pp=None, niter_PnP=10):
    # extract camera poses and focals with RANSAC-PnP
    if msk.sum() < 4:
        return None  # we need at least 4 points for PnP
    pts3d, msk = map(to_numpy, (pts3d, msk))

    H, W, THREE = pts3d.shape
    assert THREE == 3
    pixels = pixel_grid(H, W)

    if focal is None:
        S = max(W, H)
        tentative_focals = np.geomspace(S/2, S*3, 21)
    else:
        tentative_focals = [focal]

    if pp is None:
        pp = (W/2, H/2)
    else:
        pp = to_numpy(pp)

    best = 0,
    for focal in tentative_focals:
        K = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])

        success, R, T, inliers = cv2.solvePnPRansac(pts3d[msk], pixels[msk], K, None,
                                                    iterationsCount=niter_PnP, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
        if not success:
            continue

        score = len(inliers)
        if success and score > best[0]:
            best = score, R, T, focal

    if not best[0]:
        return None

    _, R, T, best_focal = best
    R = cv2.Rodrigues(R)[0]  # world to cam
    R, T = map(torch.from_numpy, (R, T))
    return best_focal, inv(sRT_to_4x4(1, R, T, device))  # cam to world


def get_known_poses(self):
    if self.has_im_poses:
        known_poses_msk = torch.tensor([not (p.requires_grad) for p in self.im_poses])
        known_poses = self.get_im_poses()
        return known_poses_msk.sum(), known_poses_msk, known_poses
    else:
        return 0, None, None


def get_known_focals(self):
    if self.has_im_poses:
        known_focal_msk = self.get_known_focal_mask()
        known_focals = self.get_focals()
        return known_focal_msk.sum(), known_focal_msk, known_focals
    else:
        return 0, None, None


def align_multiple_poses(src_poses, target_poses):
    N = len(src_poses)
    assert src_poses.shape == target_poses.shape == (N, 4, 4)

    def center_and_z(poses):
        eps = get_med_dist_between_poses(poses) / 100
        return torch.cat((poses[:, :3, 3], poses[:, :3, 3] + eps*poses[:, :3, 2]))
    R, T, s = roma.rigid_points_registration(center_and_z(src_poses), center_and_z(target_poses), compute_scaling=True)
    return s, R, T

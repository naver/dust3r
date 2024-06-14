# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dataset utilities
# --------------------------------------------------------
import numpy as np
import quaternion
import torchvision.transforms as tvf
from dust3r.utils.geometry import geotrf


def cam_to_world_from_kapture(kdata, timestamp, camera_id):
    camera_to_world = kdata.trajectories[timestamp, camera_id].inverse()
    camera_pose = np.eye(4, dtype=np.float32)
    camera_pose[:3, :3] = quaternion.as_rotation_matrix(camera_to_world.r)
    camera_pose[:3, 3] = camera_to_world.t_raw
    return camera_pose


ratios_resolutions = {
    224: {1.0: [224, 224]},
    512: {4 / 3: [512, 384], 32 / 21: [512, 336], 16 / 9: [512, 288], 2 / 1: [512, 256], 16 / 5: [512, 160]}
}


def get_HW_resolution(H, W, maxdim, patchsize=16):
    assert maxdim in ratios_resolutions, "Error, maxdim can only be 224 or 512 for now. Other maxdims not implemented yet."
    ratios_resolutions_maxdim = ratios_resolutions[maxdim]
    mindims = set([min(res) for res in ratios_resolutions_maxdim.values()])
    ratio = W / H
    ref_ratios = np.array([*(ratios_resolutions_maxdim.keys())])
    islandscape = (W >= H)
    if islandscape:
        diff = np.abs(ratio - ref_ratios)
    else:
        diff = np.abs(ratio - (1 / ref_ratios))
    selkey = ref_ratios[np.argmin(diff)]
    res = ratios_resolutions_maxdim[selkey]
    # check patchsize and make sure output resolution is a multiple of patchsize
    if isinstance(patchsize, tuple):
        assert len(patchsize) == 2 and isinstance(patchsize[0], int) and isinstance(
            patchsize[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
        assert patchsize[0] == patchsize[1], "Error, non square patches not managed"
        patchsize = patchsize[0]
    assert max(res) == maxdim
    assert min(res) in mindims
    return res[::-1] if islandscape else res  # return HW


def get_resize_function(maxdim, patch_size, H, W, is_mask=False):
    if [max(H, W), min(H, W)] in ratios_resolutions[maxdim].values():
        return lambda x: x, np.eye(3), np.eye(3)
    else:
        target_HW = get_HW_resolution(H, W, maxdim=maxdim, patchsize=patch_size)

        ratio = W / H
        target_ratio = target_HW[1] / target_HW[0]
        to_orig_crop = np.eye(3)
        to_rescaled_crop = np.eye(3)
        if abs(ratio - target_ratio) < np.finfo(np.float32).eps:
            crop_W = W
            crop_H = H
        elif ratio - target_ratio < 0:
            crop_W = W
            crop_H = int(W / target_ratio)
            to_orig_crop[1, 2] = (H - crop_H) / 2.0
            to_rescaled_crop[1, 2] = -(H - crop_H) / 2.0
        else:
            crop_W = int(H * target_ratio)
            crop_H = H
            to_orig_crop[0, 2] = (W - crop_W) / 2.0
            to_rescaled_crop[0, 2] = - (W - crop_W) / 2.0

        crop_op = tvf.CenterCrop([crop_H, crop_W])

        if is_mask:
            resize_op = tvf.Resize(size=target_HW, interpolation=tvf.InterpolationMode.NEAREST_EXACT)
        else:
            resize_op = tvf.Resize(size=target_HW)
        to_orig_resize = np.array([[crop_W / target_HW[1], 0, 0],
                                   [0, crop_H / target_HW[0], 0],
                                   [0, 0, 1]])
        to_rescaled_resize = np.array([[target_HW[1] / crop_W, 0, 0],
                                       [0, target_HW[0] / crop_H, 0],
                                       [0, 0, 1]])

        op = tvf.Compose([crop_op, resize_op])

        return op, to_rescaled_resize @ to_rescaled_crop, to_orig_crop @ to_orig_resize


def rescale_points3d(pts2d, pts3d, to_resize, HR, WR):
    # rescale pts2d as floats
    # to colmap, so that the image is in [0, D] -> [0, NewD]
    pts2d = pts2d.copy()
    pts2d[:, 0] += 0.5
    pts2d[:, 1] += 0.5

    pts2d_rescaled = geotrf(to_resize, pts2d, norm=True)

    pts2d_rescaled_int = pts2d_rescaled.copy()
    # convert back to cv2 before round [-0.5, 0.5] -> pixel 0
    pts2d_rescaled_int[:, 0] -= 0.5
    pts2d_rescaled_int[:, 1] -= 0.5
    pts2d_rescaled_int = pts2d_rescaled_int.round().astype(np.int64)

    # update valid (remove cropped regions)
    valid_rescaled = (pts2d_rescaled_int[:, 0] >= 0) & (pts2d_rescaled_int[:, 0] < WR) & (
        pts2d_rescaled_int[:, 1] >= 0) & (pts2d_rescaled_int[:, 1] < HR)

    pts2d_rescaled_int = pts2d_rescaled_int[valid_rescaled]

    # rebuild pts3d from rescaled ps2d poses
    pts3d_rescaled = np.full((HR, WR, 3), np.nan, dtype=np.float32)  # pts3d in 512 x something
    pts3d_rescaled[pts2d_rescaled_int[:, 1], pts2d_rescaled_int[:, 0]] = pts3d[valid_rescaled]

    return pts2d_rescaled, pts2d_rescaled_int, pts3d_rescaled, np.isfinite(pts3d_rescaled.sum(axis=-1))

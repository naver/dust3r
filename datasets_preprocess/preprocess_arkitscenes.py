#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Script to pre-process the arkitscenes dataset.
# Usage:
# python3 datasets_preprocess/preprocess_arkitscenes.py --arkitscenes_dir /path/to/arkitscenes --precomputed_pairs /path/to/arkitscenes_pairs
# --------------------------------------------------------
import os
import json
import os.path as osp
import decimal
import argparse
import math
from bisect import bisect_left
from PIL import Image
import numpy as np
import quaternion
from scipy import interpolate
import cv2


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arkitscenes_dir', required=True)
    parser.add_argument('--precomputed_pairs', required=True)
    parser.add_argument('--output_dir', default='data/arkitscenes_processed')
    return parser


def value_to_decimal(value, decimal_places):
    decimal.getcontext().rounding = decimal.ROUND_HALF_UP  # define rounding method
    return decimal.Decimal(str(float(value))).quantize(decimal.Decimal('1e-{}'.format(decimal_places)))


def closest(value, sorted_list):
    index = bisect_left(sorted_list, value)
    if index == 0:
        return sorted_list[0]
    elif index == len(sorted_list):
        return sorted_list[-1]
    else:
        value_before = sorted_list[index - 1]
        value_after = sorted_list[index]
        if value_after - value < value - value_before:
            return value_after
        else:
            return value_before


def get_up_vectors(pose_device_to_world):
    return np.matmul(pose_device_to_world, np.array([[0.0], [-1.0], [0.0], [0.0]]))


def get_right_vectors(pose_device_to_world):
    return np.matmul(pose_device_to_world, np.array([[1.0], [0.0], [0.0], [0.0]]))


def read_traj(traj_path):
    quaternions = []
    poses = []
    timestamps = []
    poses_p_to_w = []
    with open(traj_path) as f:
        traj_lines = f.readlines()
        for line in traj_lines:
            tokens = line.split()
            assert len(tokens) == 7
            traj_timestamp = float(tokens[0])

            timestamps_decimal_value = value_to_decimal(traj_timestamp, 3)
            timestamps.append(float(timestamps_decimal_value))  # for spline interpolation

            angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
            r_w_to_p, _ = cv2.Rodrigues(np.asarray(angle_axis))
            t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])

            pose_w_to_p = np.eye(4)
            pose_w_to_p[:3, :3] = r_w_to_p
            pose_w_to_p[:3, 3] = t_w_to_p

            pose_p_to_w = np.linalg.inv(pose_w_to_p)

            r_p_to_w_as_quat = quaternion.from_rotation_matrix(pose_p_to_w[:3, :3])
            t_p_to_w = pose_p_to_w[:3, 3]
            poses_p_to_w.append(pose_p_to_w)
            poses.append(t_p_to_w)
            quaternions.append(r_p_to_w_as_quat)
    return timestamps, poses, quaternions, poses_p_to_w


def main(rootdir, pairsdir, outdir):
    os.makedirs(outdir, exist_ok=True)

    subdirs = ['Test', 'Training']
    for subdir in subdirs:
        if not osp.isdir(osp.join(rootdir, subdir)):
            continue
        # STEP 1: list all scenes
        outsubdir = osp.join(outdir, subdir)
        os.makedirs(outsubdir, exist_ok=True)
        listfile = osp.join(pairsdir, subdir, 'scene_list.json')
        with open(listfile, 'r') as f:
            scene_dirs = json.load(f)

        valid_scenes = []
        for scene_subdir in scene_dirs:
            out_scene_subdir = osp.join(outsubdir, scene_subdir)
            os.makedirs(out_scene_subdir, exist_ok=True)

            scene_dir = osp.join(rootdir, subdir, scene_subdir)
            depth_dir = osp.join(scene_dir, 'lowres_depth')
            rgb_dir = osp.join(scene_dir, 'vga_wide')
            intrinsics_dir = osp.join(scene_dir, 'vga_wide_intrinsics')
            traj_path = osp.join(scene_dir, 'lowres_wide.traj')

            # STEP 2: read selected_pairs.npz
            selected_pairs_path = osp.join(pairsdir, subdir, scene_subdir, 'selected_pairs.npz')
            selected_npz = np.load(selected_pairs_path)
            selection, pairs = selected_npz['selection'], selected_npz['pairs']
            selected_sky_direction_scene = str(selected_npz['sky_direction_scene'][0])
            if len(selection) == 0 or len(pairs) == 0:
                # not a valid scene
                continue
            valid_scenes.append(scene_subdir)

            # STEP 3: parse the scene and export the list of valid (K, pose, rgb, depth) and convert images
            scene_metadata_path = osp.join(out_scene_subdir, 'scene_metadata.npz')
            if osp.isfile(scene_metadata_path):
                continue
            else:
                print(f'parsing {scene_subdir}')
                # loads traj
                timestamps, poses, quaternions, poses_cam_to_world = read_traj(traj_path)

                poses = np.array(poses)
                quaternions = np.array(quaternions, dtype=np.quaternion)
                quaternions = quaternion.unflip_rotors(quaternions)
                timestamps = np.array(timestamps)

                selected_images = [(basename, basename.split(".png")[0].split("_")[1]) for basename in selection]
                timestamps_selected = [float(frame_id) for _, frame_id in selected_images]

                sky_direction_scene, trajectories, intrinsics, images = convert_scene_metadata(scene_subdir,
                                                                                               intrinsics_dir,
                                                                                               timestamps,
                                                                                               quaternions,
                                                                                               poses,
                                                                                               poses_cam_to_world,
                                                                                               selected_images,
                                                                                               timestamps_selected)
                assert selected_sky_direction_scene == sky_direction_scene

                os.makedirs(os.path.join(out_scene_subdir, 'vga_wide'), exist_ok=True)
                os.makedirs(os.path.join(out_scene_subdir, 'lowres_depth'), exist_ok=True)
                assert isinstance(sky_direction_scene, str)
                for basename in images:
                    img_out = os.path.join(out_scene_subdir, 'vga_wide', basename.replace('.png', '.jpg'))
                    depth_out = os.path.join(out_scene_subdir, 'lowres_depth', basename)
                    if osp.isfile(img_out) and osp.isfile(depth_out):
                        continue

                    vga_wide_path = osp.join(rgb_dir, basename)
                    depth_path = osp.join(depth_dir, basename)

                    img = Image.open(vga_wide_path)
                    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

                    # rotate the image
                    if sky_direction_scene == 'RIGHT':
                        try:
                            img = img.transpose(Image.Transpose.ROTATE_90)
                        except Exception:
                            img = img.transpose(Image.ROTATE_90)
                        depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif sky_direction_scene == 'LEFT':
                        try:
                            img = img.transpose(Image.Transpose.ROTATE_270)
                        except Exception:
                            img = img.transpose(Image.ROTATE_270)
                        depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
                    elif sky_direction_scene == 'DOWN':
                        try:
                            img = img.transpose(Image.Transpose.ROTATE_180)
                        except Exception:
                            img = img.transpose(Image.ROTATE_180)
                        depth = cv2.rotate(depth, cv2.ROTATE_180)

                    W, H = img.size
                    if not osp.isfile(img_out):
                        img.save(img_out)

                    depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST_EXACT)
                    if not osp.isfile(depth_out):  # avoid destroying the base dataset when you mess up the paths
                        cv2.imwrite(depth_out, depth)

                # save at the end
                np.savez(scene_metadata_path,
                         trajectories=trajectories,
                         intrinsics=intrinsics,
                         images=images,
                         pairs=pairs)

        outlistfile = osp.join(outsubdir, 'scene_list.json')
        with open(outlistfile, 'w') as f:
            json.dump(valid_scenes, f)

        # STEP 5: concat all scene_metadata.npz into a single file
        scene_data = {}
        for scene_subdir in valid_scenes:
            scene_metadata_path = osp.join(outsubdir, scene_subdir, 'scene_metadata.npz')
            with np.load(scene_metadata_path) as data:
                trajectories = data['trajectories']
                intrinsics = data['intrinsics']
                images = data['images']
                pairs = data['pairs']
            scene_data[scene_subdir] = {'trajectories': trajectories,
                                        'intrinsics': intrinsics,
                                        'images': images,
                                        'pairs': pairs}
        offset = 0
        counts = []
        scenes = []
        sceneids = []
        images = []
        intrinsics = []
        trajectories = []
        pairs = []
        for scene_idx, (scene_subdir, data) in enumerate(scene_data.items()):
            num_imgs = data['images'].shape[0]
            img_pairs = data['pairs']

            scenes.append(scene_subdir)
            sceneids.extend([scene_idx] * num_imgs)

            images.append(data['images'])

            K = np.expand_dims(np.eye(3), 0).repeat(num_imgs, 0)
            K[:, 0, 0] = [fx for _, _, fx, _, _, _ in data['intrinsics']]
            K[:, 1, 1] = [fy for _, _, _, fy, _, _ in data['intrinsics']]
            K[:, 0, 2] = [hw for _, _, _, _, hw, _ in data['intrinsics']]
            K[:, 1, 2] = [hh for _, _, _, _, _, hh in data['intrinsics']]

            intrinsics.append(K)
            trajectories.append(data['trajectories'])

            # offset pairs
            img_pairs[:, 0:2] += offset
            pairs.append(img_pairs)
            counts.append(offset)

            offset += num_imgs

        images = np.concatenate(images, axis=0)
        intrinsics = np.concatenate(intrinsics, axis=0)
        trajectories = np.concatenate(trajectories, axis=0)
        pairs = np.concatenate(pairs, axis=0)
        np.savez(osp.join(outsubdir, 'all_metadata.npz'),
                 counts=counts,
                 scenes=scenes,
                 sceneids=sceneids,
                 images=images,
                 intrinsics=intrinsics,
                 trajectories=trajectories,
                 pairs=pairs)


def convert_scene_metadata(scene_subdir, intrinsics_dir,
                           timestamps, quaternions, poses, poses_cam_to_world,
                           selected_images, timestamps_selected):
    # find scene orientation
    sky_direction_scene, rotated_to_cam = find_scene_orientation(poses_cam_to_world)

    # find/compute pose for selected timestamps
    # most images have a valid timestamp / exact pose associated
    timestamps_selected = np.array(timestamps_selected)
    spline = interpolate.interp1d(timestamps, poses, kind='linear', axis=0)
    interpolated_rotations = quaternion.squad(quaternions, timestamps, timestamps_selected)
    interpolated_positions = spline(timestamps_selected)

    trajectories = []
    intrinsics = []
    images = []
    for i, (basename, frame_id) in enumerate(selected_images):
        intrinsic_fn = osp.join(intrinsics_dir, f"{scene_subdir}_{frame_id}.pincam")
        if not osp.exists(intrinsic_fn):
            intrinsic_fn = osp.join(intrinsics_dir, f"{scene_subdir}_{float(frame_id) - 0.001:.3f}.pincam")
        if not osp.exists(intrinsic_fn):
            intrinsic_fn = osp.join(intrinsics_dir, f"{scene_subdir}_{float(frame_id) + 0.001:.3f}.pincam")
        assert osp.exists(intrinsic_fn)
        w, h, fx, fy, hw, hh = np.loadtxt(intrinsic_fn)  # PINHOLE

        pose = np.eye(4)
        pose[:3, :3] = quaternion.as_rotation_matrix(interpolated_rotations[i])
        pose[:3, 3] = interpolated_positions[i]

        images.append(basename)
        if sky_direction_scene == 'RIGHT' or sky_direction_scene == 'LEFT':
            intrinsics.append([h, w, fy, fx, hh, hw])  # swapped intrinsics
        else:
            intrinsics.append([w, h, fx, fy, hw, hh])
        trajectories.append(pose  @ rotated_to_cam)  # pose_cam_to_world @ rotated_to_cam = rotated(cam) to world

    return sky_direction_scene, trajectories, intrinsics, images


def find_scene_orientation(poses_cam_to_world):
    if len(poses_cam_to_world) > 0:
        up_vector = sum(get_up_vectors(p) for p in poses_cam_to_world) / len(poses_cam_to_world)
        right_vector = sum(get_right_vectors(p) for p in poses_cam_to_world) / len(poses_cam_to_world)
        up_world = np.array([[0.0], [0.0], [1.0], [0.0]])
    else:
        up_vector = np.array([[0.0], [-1.0], [0.0], [0.0]])
        right_vector = np.array([[1.0], [0.0], [0.0], [0.0]])
        up_world = np.array([[0.0], [0.0], [1.0], [0.0]])

    # value between 0, 180
    device_up_to_world_up_angle = np.arccos(np.clip(np.dot(np.transpose(up_world),
                                                           up_vector), -1.0, 1.0)).item() * 180.0 / np.pi
    device_right_to_world_up_angle = np.arccos(np.clip(np.dot(np.transpose(up_world),
                                                              right_vector), -1.0, 1.0)).item() * 180.0 / np.pi

    up_closest_to_90 = abs(device_up_to_world_up_angle - 90.0) < abs(device_right_to_world_up_angle - 90.0)
    if up_closest_to_90:
        assert abs(device_up_to_world_up_angle - 90.0) < 45.0
        # LEFT
        if device_right_to_world_up_angle > 90.0:
            sky_direction_scene = 'LEFT'
            cam_to_rotated_q = quaternion.from_rotation_vector([0.0, 0.0, math.pi / 2.0])
        else:
            # note that in metadata.csv RIGHT does not exist, but again it's not accurate...
            # well, turns out there are scenes oriented like this
            # for example Training/41124801
            sky_direction_scene = 'RIGHT'
            cam_to_rotated_q = quaternion.from_rotation_vector([0.0, 0.0, -math.pi / 2.0])
    else:
        # right is close to 90
        assert abs(device_right_to_world_up_angle - 90.0) < 45.0
        if device_up_to_world_up_angle > 90.0:
            sky_direction_scene = 'DOWN'
            cam_to_rotated_q = quaternion.from_rotation_vector([0.0, 0.0, math.pi])
        else:
            sky_direction_scene = 'UP'
            cam_to_rotated_q = quaternion.quaternion(1, 0, 0, 0)
    cam_to_rotated = np.eye(4)
    cam_to_rotated[:3, :3] = quaternion.as_rotation_matrix(cam_to_rotated_q)
    rotated_to_cam = np.linalg.inv(cam_to_rotated)
    return sky_direction_scene, rotated_to_cam


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args.arkitscenes_dir, args.precomputed_pairs, args.output_dir)

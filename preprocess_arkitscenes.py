from pathlib import Path
import json
import math
import random
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import cv2
from skimage.io import imread
from skimage.transform import resize
from PIL import Image

from dust3r.datasets.utils.sampling import get_camera_visual_cone, cone_IoU, create_cone_vis
from dust3r.datasets.utils import cropping
from dust3r.datasets.utils.sampling import compute_camera_iou

def decide_pose(pose):
    """
    Args:
        pose: np.array (4, 4)
    Returns:
        index: int (0, 1, 2, 3)
        for upright, left, upside-down and right
    """
    z_vec = pose[2, :3]
    z_orien = np.array(
        [
            [0.0, -1.0, 0.0],  # upright
            [-1.0, 0.0, 0.0],  # left
            [0.0, 1.0, 0.0],  # upside-down
            [1.0, 0.0, 0.0],
        ]  # right
    )
    corr = np.matmul(z_orien, z_vec)
    corr_max = np.argmax(corr)
    return corr_max

def rotate_pose(im, rot_index):
    """
    Args:
        im: (m, n)
    """
    h, w, d = im.shape
    if d == 3:
        if rot_index == 0:
            new_im = im
        elif rot_index == 1:
            new_im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif rot_index == 2:
            new_im = cv2.rotate(im, cv2.ROTATE_180)
        elif rot_index == 3:
            new_im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return new_im

def opencv_from_cameras_projection(R, T, focal, p0, image_size):
    R = torch.from_numpy(R)[None, :, :]
    T = torch.from_numpy(T)[None, :]
    focal = torch.from_numpy(focal)[None, :]
    p0 = torch.from_numpy(p0)[None, :]
    image_size = torch.from_numpy(image_size)[None, :]
    R_pytorch3d = R.clone()
    T_pytorch3d = T.clone()
    focal_pytorch3d = focal
    p0_pytorch3d = p0
    tvec = T_pytorch3d
    R = R_pytorch3d.permute(0, 2, 1)

    focal_length = focal_pytorch3d
    principal_point = p0_pytorch3d

    camera_matrix = torch.zeros_like(R)
    camera_matrix[:, :2, 2] = principal_point
    camera_matrix[:, 2, 2] = 1.0
    camera_matrix[:, 0, 0] = focal_length[:, 0]
    camera_matrix[:, 1, 1] = focal_length[:, 1]
    return R[0], tvec[0], camera_matrix[0]

def crop_and_resize_dataset(input_depthmap, input_rgb_image, camera_intrinsics, img_size=256):
    input_mask = np.ones(input_depthmap.shape)
    depth_mask = np.stack((input_depthmap, input_mask), axis=-1)
    H, W = input_depthmap.shape
    cx, cy = camera_intrinsics[:2, 2].round().astype(int)
    # print(H,W, cx, cy)
    min_margin_x = min(cx, W-cx)
    min_margin_y = min(cy, H-cy)
    # print(f"min margin x: {min_margin_x} y: {min_margin_y}")

    # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
    l, t = cx - min_margin_x, cy - min_margin_y
    r, b = cx + min_margin_x, cy + min_margin_y
    crop_bbox = (l, t, r, b)
    input_rgb_image, depth_mask, input_camera_intrinsics = cropping.crop_image_depthmap(
        input_rgb_image, depth_mask, camera_intrinsics, crop_bbox)

    # try to set the lower dimension to img_size * 3/4 -> img_size=512 => 384
    scale_final = ((img_size * 3 // 4) / min(H, W)) + 1e-8
    output_resolution = np.floor(np.array([W, H]) * scale_final).astype(int)
    # print(f"scaling output resolution: {output_resolution}")
    if max(output_resolution) < img_size:
        # let's put the max dimension to img_size
        scale_final = (img_size / max(H, W)) + 1e-8
        output_resolution = np.floor(np.array([W, H]) * scale_final).astype(int)

    input_rgb_image, depth_mask, input_camera_intrinsics = cropping.rescale_image_depthmap(
        input_rgb_image, depth_mask, input_camera_intrinsics, output_resolution)
    input_depthmap = depth_mask[:, :, 0]
    input_mask = depth_mask[:, :, 1]
    return input_depthmap, input_rgb_image, input_mask, input_camera_intrinsics

def prepare_sequences(
    scene_df: pd.DataFrame,
    input_dir: str,
    split: str = "train",
    frame_subsample_rate=0.5,
    pair_subsample_rate=0.25,
    frame_zfill=6,
    min_frame_desync=20,
):
    output_dir = Path(str(input_dir) + "_processed")
    dataset = "raw"
    img_subdir = "lowres_wide"
    depth_subdir = "lowres_depth"
    traj_fname = "lowres_wide.traj"

    for vid_id,row in scene_df.iterrows():
        if vid_id != 40753679:
            continue
        print(f"processing scene {vid_id}")
        subdir_split = row["fold"]
        if split not in subdir_split.lower():
            continue
        vid_dir = input_dir / f"{dataset}/{subdir_split}/{vid_id}"
        out_vid_dir = output_dir / f"{dataset}/{vid_id}"
        if not vid_dir.exists():
            print(f"no data found in {vid_dir}")
            continue
        traj_path = vid_dir / traj_fname
        if not traj_path.exists():
            continue
        timestamps = []
        extrinsics = []
        with open(traj_path, "rt") as infile:
            for line in infile:
                linelist = line.split()
                timestamps.append(float(linelist[0]))
                extrinsics.append([float(v) for v in linelist[1:]])
        extrinsics = np.array(extrinsics)
        extrinsics = pd.DataFrame(extrinsics, columns=["r1", "r2", "r3", "t1", "t2", "t3"], index=pd.Index(timestamps))
        fids_fnames = list(sorted([(i,v.stem) for i,v in enumerate(sorted((vid_dir / img_subdir).glob("*.png")))]))
        sample_fids_fnames = sorted(random.sample(fids_fnames, int(round(len(fids_fnames) * frame_subsample_rate))))
        # sample_fids_fnames = sorted([fids_fnames[349], fids_fnames[559]])
        print(
            f"sampling {len(sample_fids_fnames)} of {len(fids_fnames)} total images"
            f" ({len(sample_fids_fnames)/len(fids_fnames)*100:.0f}%)"
        )
        intrinsics_by_frame = dict()
        extrinsics_by_frame = dict()
        sequence_data = defaultdict(lambda: [])
        for frame_idx,fname in sample_fids_fnames:
            new_fname = f"frame{str(frame_idx).zfill(frame_zfill)}"
            save_img_path = out_vid_dir / f"images/{new_fname}.jpg"
            save_meta_path = out_vid_dir / f"images/{new_fname}.npz"
            save_depth_path = out_vid_dir / f"depths/{new_fname}.jpg.geometric.png"
            save_mask_path = out_vid_dir / f"masks/{new_fname}.png"
            
            ### Check frame desync
            fname_ts = float(fname.split("_")[-1])
            diffs = (extrinsics.index.values - fname_ts) * 1000
            closest_idx = np.argmin(np.abs(diffs))
            diff = diffs[closest_idx]
            if np.abs(diff) > min_frame_desync:
                print(f"frame {frame_idx} bad frame desync: {np.abs(diff)}")
                continue

            if not all([save_img_path.exists(), save_meta_path.exists(), save_depth_path.exists(), save_mask_path.exists()]):
                intrinsics_path = vid_dir / f"{img_subdir}_intrinsics" / f"{fname}.pincam"
                depth_path = vid_dir / depth_subdir / f"{fname}.png"
                image_path = vid_dir / img_subdir / f"{fname}.png"
                if not all([intrinsics_path.exists(), depth_path.exists(), image_path.exists()]):
                    print(f"missing input data for {fname}")
                    continue
                print(f"frame index: {frame_idx} extrinsics index: {closest_idx} (frame desync: {diff:.0f} ms)")
                cur_extrinsics = extrinsics.iloc[closest_idx].values
                R, _ = cv2.Rodrigues(cur_extrinsics[:3])
                T = cur_extrinsics[3:]
                with open(intrinsics_path, "rt") as infile:
                    lines = list([l for l in infile])
                assert len(lines) == 1
                w, h, fx, fy, px, py = [float(v) for v in lines[0].split()]
                focal = np.array([fx, fy])
                p0 = np.array([px, py])
                image_size = np.array([h, w])
                _, _, intrinsics = opencv_from_cameras_projection(R, T, focal, p0, image_size)
                # print(f"image height: {h} width: {w} focal: {focal} principal: {p0}")
                # print(f"camera metadata (R: {R.shape}, T: {T.shape}, intrinsics: {intrinsics.shape}):")
                depth_map = imread(depth_path).astype(np.float32)
                # convert milimeters to meters
                depth_map /= 1000
                image = imread(image_path)
                depth_map = resize(depth_map, image.shape[:2], anti_aliasing=True, preserve_range=True)
                image = Image.fromarray(image)
                depth_map, image, mask, intrinsics = crop_and_resize_dataset(depth_map, image, intrinsics.numpy())
                # print(f"\nfinal outputs:")
                # print(depth_map.shape, image.size, mask.shape, intrinsics.shape)

                # generate and adjust camera pose
                camera_pose = np.eye(4, dtype=np.float32)
                camera_pose[:3, :3] = R
                camera_pose[:3,  3] = T
                camera_pose = np.linalg.inv(camera_pose)

                save_img_path.parent.mkdir(parents=True, exist_ok=True)
                save_depth_path.parent.mkdir(parents=True, exist_ok=True)
                save_mask_path.parent.mkdir(parents=True, exist_ok=True)

                image.save(str(save_img_path))
                scaled_depth_map = (depth_map / np.max(depth_map) * 65535).astype(np.uint16)
                cv2.imwrite(str(save_depth_path), scaled_depth_map)
                cv2.imwrite(str(save_mask_path), (mask * 255).astype(np.uint8))
                np.savez(
                    save_meta_path,
                    camera_intrinsics=intrinsics,
                    camera_pose=camera_pose,
                    maximum_depth=np.max(depth_map)
                )
                print(f"successfully processed data for {new_fname} ({intrinsics.shape},{camera_pose.shape},{image.size},{depth_map.shape})")
            else:
                print(f"already processed data for {new_fname}")
            if all(
                [save_img_path.exists(), save_meta_path.exists(), save_depth_path.exists(), save_mask_path.exists()]
            ):
                sequence_data[vid_id].append(frame_idx)
                meta = np.load(save_meta_path)
                intrinsics_by_frame[frame_idx] = meta["camera_intrinsics"]
                extrinsics_by_frame[frame_idx] = meta["camera_pose"]
                print(intrinsics_by_frame[frame_idx])
                print(extrinsics_by_frame[frame_idx])
            else:
                print(f"unable to process frame {frame_idx}: {fname}")

        cur_sequence_pairs = []
        all_fids = sorted(intrinsics_by_frame.keys())
        sequence_pairs = {}
        total_pair_counts = 0
        print(f"sampling pairs out of {len(all_fids)} final frames")
        for i in range(len(all_fids)):
            fid1 = all_fids[i]
            ints1 = intrinsics_by_frame[fid1]
            exts1 = extrinsics_by_frame[fid1]
            all_second_frames = range(i+1, len(all_fids))
            sample_second_frames = random.sample(all_second_frames, int(round(len(all_second_frames) * pair_subsample_rate)))
            # print(f"frame {fid1}: sampled {len(sample_second_frames)} of {len(all_second_frames)} potential pair frames")
            for j in sample_second_frames:
                fid2 = all_fids[j]
                ints2 = intrinsics_by_frame[fid2]
                exts2 = extrinsics_by_frame[fid2]
                img_height1 = int(round(ints1[1][2] * 2))
                img_height2 = int(round(ints2[1][2] * 2))
                iou, mesh = compute_camera_iou(exts1, exts2, ints1, ints2, height=25, num_samples=100)
                if iou > 0:
                    print(fid1, fid2, iou)
                    cur_sequence_pairs.append((fid1, fid2, iou))
                total_pair_counts += 1
        print(f"{len(cur_sequence_pairs)} valid pairs of {total_pair_counts} potential")
        sequence_pairs[vid_id] = cur_sequence_pairs

    seq_data_path = output_dir / dataset / "selected_seqs_train.json"
    with open(seq_data_path, "wt") as outfile:
        json.dump(sequence_data, outfile)

    all_seq_data_path = output_dir / "selected_seqs_train.json"
    with open(all_seq_data_path, "wt") as outfile:
        json.dump({dataset: sequence_data}, outfile)

    all_seq_pairs_path = output_dir / "selected_pairs_train.json"
    with open(all_seq_pairs_path, "wt") as outfile:
        json.dump({dataset: sequence_pairs}, outfile)

if __name__ == "__main__":
    input_dir = Path("/Users/brad/workspace/neuro3d/data/ARKitScenes_small")
    df = pd.read_csv(input_dir / "raw/metadata.csv", index_col="video_id")
    print(df.shape)

    prepare_sequences(
        scene_df=df,
        input_dir=input_dir,
        split="train",
        frame_subsample_rate=0.167, # equivalent of 10 fps (same as hi-res data)
        pair_subsample_rate=0.05, # get sampling down to about 800 pairs per scene
        min_frame_desync=1,
    )
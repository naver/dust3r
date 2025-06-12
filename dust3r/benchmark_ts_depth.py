import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
import pandas as pd
from dataclasses import dataclass
import json
from typing import List, Dict
import os
import argparse
import torch
import logging
import time
from omegaconf import OmegaConf
import random
import numpy as np
import matplotlib.pyplot as plt
import imageio

from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder
from scripts.deserialize_depth_dataset import Boto3ResourceManager, deserialize_and_download_image, deserialize_and_download_tensor
from Utils import set_logging_format, set_seed, vis_disparity


def load_model(args):
    """Loads the stereo model and checkpoint.

    Args:
        args: Command-line arguments.

    Returns:
        model: The loaded stereo model.
    """
    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        if k not in ['left_file', 'right_file']: # Avoid overwriting constructed paths
             cfg[k] = args.__dict__[k]
    current_args = OmegaConf.create(cfg) # Use a different name to avoid confusion
    logging.info(f"args for model loading:\n{current_args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")

    model = FoundationStereo(current_args)

    ckpt = torch.load(ckpt_dir)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])

    model.cuda()
    model.eval()
    return model


@dataclass
class DepthData:
    dataset_creator: str
    camera_names: List[str]
    item_id: int
    split: str
    image_paths: Dict[str, str]
    depth_map_paths: Dict[str, str]
    normal_map_paths: Dict[str, str]
    visible_mask_paths: Dict[str, str]
    world_from_camera_transforms_path: str
    camera_intrinsics_path: str

    @classmethod
    def from_row(cls, row):
        return cls(
            dataset_creator=row[0],
            camera_names=list(row[1]),
            item_id=row[2],
            split=row[3],
            image_paths=json.loads(row[4]),
            depth_map_paths=json.loads(row[5]),
            normal_map_paths=json.loads(row[6]),
            visible_mask_paths=json.loads(row[7]),
            world_from_camera_transforms_path=row[8],
            camera_intrinsics_path=row[9],
        )


def deserialize_data(data: DepthData, resource_manager: Boto3ResourceManager, args):
    """Deserialize all the data we need for a single benchmark item."""
    camera_ids = list(data.image_paths.keys())
    if len(camera_ids) < 2:
        raise ValueError(
            f"Need at least two images for inference, but got {len(camera_ids)}.")

    cam1_id, cam2_id = random.sample(camera_ids, 2)
    logging.info(f"Randomly selected cameras: {cam1_id}, {cam2_id}")

    # It's conventional to use bit_depth=8 for RGB images.
    print(f"data.image_paths[cam1_id]: {data.image_paths[cam1_id]}")
    print(f"data.image_paths[cam2_id]: {data.image_paths[cam2_id]}")
    img1 = deserialize_and_download_image(
        data.image_paths[cam1_id], bit_depth=8, resource_manager=resource_manager, dtype=torch.float32) * 255
    img2 = deserialize_and_download_image(
        data.image_paths[cam2_id], bit_depth=8, resource_manager=resource_manager, dtype=torch.float32) * 255
    img1 = img1.cuda()
    img2 = img2.cuda()

    
    # With imageio
    
    # img1 = imageio.imread("data/20250611171250_left.png")
    # img2 = imageio.imread("data/20250611171250_right.png")
    # if img1.shape[-1] == 4:
    #     img1 = img1[..., :3]
    # if img2.shape[-1] == 4:
    #     img2 = img2[..., :3]
    # img1 = torch.as_tensor(img1).cuda().float().permute(2, 0, 1)
    # img2 = torch.as_tensor(img2).cuda().float().permute(2, 0, 1)
    # print(f"After img1.shape: {img1.shape} {img1.dtype=} {img1.min()=} {img1.max()=}")
    # print(f"After img2.shape: {img2.shape} {img2.dtype=} {img2.min()=} {img2.max()=}")
    
    
    depth_gt = deserialize_and_download_tensor(
        data.depth_map_paths[cam1_id], resource_manager=resource_manager)
    print(f"GT depth image max : {depth_gt.max()}, min: {depth_gt.min()}")
    print(f"img1.shape before crop: {img1.shape}")

    all_intrinsics = deserialize_and_download_tensor(
        data.camera_intrinsics_path, resource_manager=resource_manager)
    
    cam1_idx = data.camera_names.index(cam1_id)
    intrinsics = all_intrinsics[cam1_idx]

    all_world_from_camera_transforms = deserialize_and_download_tensor(
        data.world_from_camera_transforms_path, resource_manager=resource_manager)
    cam2_idx = data.camera_names.index(cam2_id)
    
    transform1 = all_world_from_camera_transforms[cam1_idx]
    transform2 = all_world_from_camera_transforms[cam2_idx]
    
    # The translation vector is the last column of the 4x4 matrix
    t1 = transform1[:3, 3]
    t2 = transform2[:3, 3]
    
    baseline = torch.linalg.norm(t1 - t2).item()

    print(f"Original intrinsics: \n{intrinsics}")
    refactored_intrinsics = intrinsics.clone()

    C, H, W = img1.shape[-3:]
    target_h, target_w = 1200, 1600

    if H > target_h:
        y_offset = (H - target_h) // 2
        img1 = img1[..., y_offset:y_offset + target_h, :]
        img2 = img2[..., y_offset:y_offset + target_h, :]
        depth_gt = depth_gt[..., y_offset:y_offset + target_h, :]
        # adjust intrinsics. cy is usually intrinsics[..., 1, 2]
        refactored_intrinsics[..., 1, 2] -= y_offset

    if W > target_w:
        x_offset = (W - target_w) // 2
        img1 = img1[..., :, x_offset:x_offset + target_w]
        img2 = img2[..., :, x_offset:x_offset + target_w]
        depth_gt = depth_gt[..., :, x_offset:x_offset + target_w]
        # adjust intrinsics. cx is usually intrinsics[..., 0, 2]
        refactored_intrinsics[..., 0, 2] -= x_offset
    
    print(f"Refactored intrinsics: \n{refactored_intrinsics}")
    print(f"img1.shape after crop: {img1.shape}")


    # The model expects a batch dimension
    return img1[None], img2[None], refactored_intrinsics, depth_gt, baseline


def run_inference(model, img0, img1, args):
    """Runs inference on a pair of image tensors.

    Args:
        model: The stereo model.
        img0_torch: Left image tensor.
        img1_torch: Right image tensor.
        args: Command-line arguments.

    Returns:
        disp: The disparity map.
        inference_time: The time taken for inference.
    """
    img0_torch = img0.clone()
    img1_torch = img1.clone()
    H, W = img0_torch.shape[2:]
    padder = InputPadder(img0_torch.shape, divis_by=32, force_square=False)
    img0_padded, img1_padded = padder.pad(img0_torch, img1_torch)

    start_time = time.time()
    with torch.cuda.amp.autocast(True):
        if not args.hiera:
            disp = model.forward(img0_padded, img1_padded,
                                 iters=args.valid_iters, test_mode=True, low_memory=False)
        else:
            disp = model.run_hierachical(
                img0_padded, img1_padded, iters=args.valid_iters, test_mode=True, small_ratio=0.5, low_memory=True)
    inference_time = time.time() - start_time

    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W)
    return disp, inference_time


def compare_and_visualize(img1, pred_depth, depth_gt, item_id, out_dir):
    """
    Creates a 2x2 plot comparing predicted depth with ground truth.
    Saves the plot to a file.
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    if isinstance(depth_gt, torch.Tensor):
        depth_gt = depth_gt.cpu().numpy()

    # Squeeze out channel dimension if it exists
    if depth_gt.ndim == 3 and depth_gt.shape[0] == 1:
        depth_gt = np.squeeze(depth_gt, axis=0)
    
    depth_diff = np.abs(pred_depth - depth_gt)

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(f"Item ID: {item_id}")

    im = axes[0, 0].imshow(img1)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    im = axes[0, 1].imshow(depth_diff, cmap='hot')
    axes[0, 1].set_title("Depth Difference (abs)")
    axes[0, 1].axis('off')
    fig.colorbar(im, ax=axes[0, 1])

    # Determine shared color range for depth plots
    valid_pred_depth = pred_depth[np.isfinite(pred_depth)]
    valid_depth_gt = depth_gt[np.isfinite(depth_gt)]
    vmin, vmax = None, None
    if valid_pred_depth.size > 0 and valid_depth_gt.size > 0:
        vmin = min(np.min(valid_pred_depth), np.min(valid_depth_gt))
        vmax = max(np.max(valid_pred_depth), np.max(valid_depth_gt))
    else:
        logging.warning(f"Could not determine a valid color range for item {item_id}. Using separate color bars.")

    im = axes[1, 0].imshow(depth_gt, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Ground Truth Depth")
    axes[1, 0].axis('off')
    fig.colorbar(im, ax=axes[1, 0])
    
    im = axes[1, 1].imshow(pred_depth, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("Predicted Depth")
    axes[1, 1].axis('off')
    fig.colorbar(im, ax=axes[1, 1])
    
    # Add side-by-side histograms
    valid_pred_flat = valid_pred_depth.flatten()
    valid_gt_flat = valid_depth_gt.flatten()
    
    if valid_pred_flat.size > 0 and valid_gt_flat.size > 0:
        all_valid_depths = np.concatenate([valid_pred_flat, valid_gt_flat])
        # Use percentiles to avoid extreme outliers skewing the histogram range
        bins = np.linspace(np.percentile(all_valid_depths, 1), np.percentile(all_valid_depths, 99), 100)
        
        axes[2, 0].hist(valid_gt_flat, bins=bins, color='blue', alpha=0.7)
        axes[2, 0].set_title("Ground Truth Depth Histogram")
        axes[2, 0].set_xlabel("Depth")
        axes[2, 0].set_ylabel("Frequency")

        axes[2, 1].hist(valid_pred_flat, bins=bins, color='green', alpha=0.7)
        axes[2, 1].set_title("Predicted Depth Histogram")
        axes[2, 1].set_xlabel("Depth")
        axes[2, 1].sharey(axes[2, 0])  # Share y-axis for better comparison
    else:
        axes[2, 0].text(0.5, 0.5, "No valid GT data for histogram", ha='center', va='center')
        axes[2, 0].axis('off')
        axes[2, 1].text(0.5, 0.5, "No valid Pred data for histogram", ha='center', va='center')
        axes[2, 1].axis('off')

    plt.tight_layout()
    output_path = os.path.join(out_dir, f"1_depth_comparison.png")
    plt.savefig(output_path)
    logging.info(f"Saved comparison plot to {output_path}")
    plt.close(fig)


def main(args):
    model = load_model(args)
    resource_manager = Boto3ResourceManager()

    def data_fn():
        df = pd.read_parquet(args.meta_data_path)
        for i in range(len(df)):
            yield DepthData.from_row(df.iloc[2])

    for data in data_fn():
        logging.info(f"Processing item {data.item_id}")
        img1, img2, intrinsics, depth_gt, baseline = deserialize_data(
            data, resource_manager, args)
        disp, inference_time = run_inference(model, img1, img2, args)

        vis = vis_disparity(disp)
        imageio.imwrite(os.path.join(args.out_dir, f"{data.item_id}_disparity_vis.png"), vis)
        print(f"baseline: {baseline} {intrinsics=} {disp.max()=} {disp.min()=}")
        focal_length = intrinsics[0, 0]
        pred_depth = (baseline * focal_length.item()) / (disp + 1e-6)
        print(f"Predicted depth image max: {pred_depth.max()}, min: {pred_depth.min()}")
        
        if isinstance(pred_depth, torch.Tensor):
            pred_depth = pred_depth.cpu().numpy()

        compare_and_visualize(img1, pred_depth, depth_gt, data.item_id, args.out_dir)

        logging.info(
            f"Inference time: {inference_time:.4f}s, Disparity map shape: {disp.shape}")

        # We break here for now to only test one item.
        break


if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_data_path', default="metadata/depth_live_1724981057", type=str, help='path to metadata parquet file')
    parser.add_argument('--basename_dir', default=f'{code_dir}/../data/', type=str, help='directory of input images, e.g. xxx_left/right.png')
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
    parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str, help='the directory to save results')
    parser.add_argument('--scale', default=1, type=float,
                        help='downsize the image by scale, must be <=1')
    parser.add_argument('--hiera', default=0, type=int,
                        help='hierarchical inference (only needed for high-resolution images (>1K))')
    parser.add_argument('--z_far', default=10, type=float,
                        help='max depth to clip in point cloud')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
    parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
    parser.add_argument('--denoise_cloud', type=int, default=0, help='whether to denoise the point cloud')
    parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
    parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
    args = parser.parse_args()

    print("Starting test...")
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    main(args)
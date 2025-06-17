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
import random
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image

# dust3r imports
from dust3r.model import AsymmetricCroCo3DStereo, load_model as load_dust3r_model
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.image import load_images, ImgNorm
from dust3r.deserialize_depth_dataset import Boto3ResourceManager, deserialize_and_download_image, deserialize_and_download_tensor
from dust3r.demo import get_3D_model_from_scene


def load_model(args):
    """Loads the DUSt3R model."""
    if args.weights:
        model = load_dust3r_model(args.weights, args.device)
    elif args.model_name:
        model = AsymmetricCroCo3DStereo.from_pretrained(args.model_name).to(args.device)
    else:
        raise ValueError("Either --model_name or --weights must be provided.")
    logging.info(f"Loaded DUSt3R model on {args.device}")
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
    depth_gt = depth_gt
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
    
    # Print ground truth camera poses
    print("\nGround Truth Camera Poses:")
    print("Camera 1 (Reference):")
    print(transform1)
    print("\nCamera 2:")
    print(transform2)
    
    
    # The translation vector is the last column of the 4x4 matrix
    t1 = transform1[:3, 3]
    t2 = transform2[:3, 3]
    
    gt_baseline = torch.linalg.norm(t1 - t2).item()
    refactored_intrinsics = intrinsics.clone()
    print(f"Original intrinsics: \n{intrinsics}")
    print(f"GT Baseline: {gt_baseline}")
    C, H, W = img1.shape[-3:]
    # dust3r works well with smaller images, let's not crop to a large size
    # target_h, target_w = 1200, 1600

    # if H > target_h:
    #     y_offset = (H - target_h) // 2
    #     img1 = img1[..., y_offset:y_offset + target_h, :]
    #     img2 = img2[..., y_offset:y_offset + target_h, :]
    #     depth_gt = depth_gt[..., y_offset:y_offset + target_h, :]
    #     # adjust intrinsics. cy is usually intrinsics[..., 1, 2]
    #     refactored_intrinsics[..., 1, 2] -= y_offset

    # if W > target_w:
    #     x_offset = (W - target_w) // 2
    #     img1 = img1[..., :, x_offset:x_offset + target_w]
    #     img2 = img2[..., :, x_offset:x_offset + target_w]
    #     depth_gt = depth_gt[..., :, x_offset:x_offset + target_w]
    #     # adjust intrinsics. cx is usually intrinsics[..., 0, 2]
    #     refactored_intrinsics[..., 0, 2] -= x_offset


    # The model expects a batch dimension
    return img1[None], img2[None], refactored_intrinsics, depth_gt, gt_baseline, transform1, transform2


def _resize_image(image_data, size):
    """Helper to resize image and adjust focals, inspired by dust3r.utils.image"""
    rgb = image_data['rgb']
    old_h, old_w = rgb.shape[:2]
    
    if isinstance(size, int):
        new_w, new_h = size, size
    else:
        new_w, new_h = size

    pil_img = Image.fromarray(rgb)
    pil_img_resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    resized_rgb = np.array(pil_img_resized)
    
    fx, fy = image_data['focals']
    new_fx = fx * new_w / old_w
    new_fy = fy * new_h / old_h
    
    return {'rgb': resized_rgb, 'focals': (new_fx, new_fy), 'path': image_data['path']}


def prepare_image_for_dust3r(img_tensor, size, idx=0):
    """Prepare image tensor for dust3r input format.
    
    Args:
        img_tensor: Input tensor (1, C, H, W)
        size: Target size for resizing
        idx: Index of the image in the sequence
        
    Returns:
        dict: Image data in dust3r format with img, true_shape, idx, and instance
    """
    # Convert to numpy and permute to HWC
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    
    # Convert to PIL Image for resizing
    pil_img = Image.fromarray(img_np)
    W1, H1 = pil_img.size
    
    # Resize according to dust3r's logic
    if size == 224:
        # resize short side to 224 (then crop)
        pil_img = _resize_pil_image(pil_img, round(size * max(W1/H1, H1/W1)))
    else:
        # resize long side to 512
        pil_img = _resize_pil_image(pil_img, size)
    
    # Center crop
    W, H = pil_img.size
    cx, cy = W//2, H//2
    if size == 224:
        half = min(cx, cy)
        pil_img = pil_img.crop((cx-half, cy-half, cx+half, cy+half))
    else:
        halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
        if W == H:  # if square
            halfh = 3*halfw/4
        pil_img = pil_img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
    
    # Convert to dust3r format
    img_norm = ImgNorm(pil_img)
    return {
        'img': img_norm[None],  # Remove [None] as it's handled by collate_with_cat
        'true_shape': np.int32([pil_img.size[::-1]]),
        'idx': idx,
        'instance': str(idx)
    }


def _resize_pil_image(img, size):
    """Resize PIL image maintaining aspect ratio."""
    W, H = img.size
    if W > H:
        new_W = size
        new_H = int(H * size / W)
    else:
        new_H = size
        new_W = int(W * size / H)
    return img.resize((new_W, new_H), Image.Resampling.LANCZOS)


def find_optimal_scale(pred_depth, gt_depth):
    """Find the optimal scale factor between predicted and ground truth depth.
    
    Args:
        pred_depth: Predicted depth map (numpy array or torch tensor)
        gt_depth: Ground truth depth map (numpy array or torch tensor)
        
    Returns:
        scale: Optimal scale factor
        error: Mean absolute error after scaling
    """
    # Convert to numpy if needed
    if torch.is_tensor(pred_depth):
        pred_depth = pred_depth.cpu().numpy()
    if torch.is_tensor(gt_depth):
        gt_depth = gt_depth.cpu().numpy()
    
    # Remove invalid values
    valid_mask = (gt_depth > 0) & np.isfinite(gt_depth) & np.isfinite(pred_depth)
    if not np.any(valid_mask):
        return 1.0, float('inf')
    
    # Compute scale using median ratio
    ratios = gt_depth[valid_mask] / (pred_depth[valid_mask] + 1e-6)
    scale = np.median(ratios)
    
    # Compute error after scaling
    scaled_pred = pred_depth * scale
    error = np.mean(np.abs(scaled_pred[valid_mask] - gt_depth[valid_mask]))
    
    return scale, error


def run_dust3r_inference(model, img1, img2, intrinsics, args, gt_pose1=None, gt_pose2=None, niter=300, schedule='cosine', lr=0.01):
    """Runs inference on a pair of image tensors using DUSt3R.

    Args:
        model: The DUSt3R model.
        img1: Left image tensor (1, C, H, W).
        img2: Right image tensor (1, C, H, W).
        intrinsics: Camera intrinsics tensor.
        args: Command-line arguments.
        gt_pose1: Ground truth pose for first camera (4x4 matrix)
        gt_pose2: Ground truth pose for second camera (4x4 matrix)

    Returns:
        pred_depth: The predicted depth map (H, W).
        inference_time: The time taken for inference.
        pred_baseline: The predicted baseline between cameras
    """
    # Prepare images for dust3r
    print(f"img1.shape: {img1.shape}")
    img1_data = prepare_image_for_dust3r(img1, args.image_size, idx=0)
    img2_data = prepare_image_for_dust3r(img2, args.image_size, idx=1)
    # print(f"img1_data: {img1_data}")
    
    
    # Get focal lengths from intrinsics
    focals = (intrinsics[0, 0].item(), intrinsics[1, 1].item())
    
    gt_poses = [gt_pose1, gt_pose2]

    # Create list of images in dust3r format
    loaded_imgs = [img1_data, img2_data]
    pairs = make_pairs(loaded_imgs, prefilter=None, symmetrize=True)

    start_time = time.time()
    with torch.cuda.amp.autocast(True):
        output = inference(pairs, model, args.device, batch_size=1)
    inference_time = time.time() - start_time
    
    # Enable gradients for optimization
    torch.autograd.set_grad_enabled(True)
    
    scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.ModularPointCloudOptimizer, optimize_pp=True)
    
    scene.preset_pose([pose for pose in gt_poses], [True, True])
    scene.preset_focal([focals[0], focals[1]], [True, True])
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    
    # Disable gradients after optimization
    torch.autograd.set_grad_enabled(False)
    
    depth_maps = to_numpy(scene.get_depthmaps())
    pred_depth = depth_maps[0]  # Depth for the first image
    
    # Get predicted camera poses
    pred_poses = scene.get_im_poses()
    pred_pose1 = pred_poses[0]  # First camera pose
    pred_pose2 = pred_poses[1]  # Second camera pose
    
    # Print predicted camera poses
    print("\nPredicted Camera Poses:")
    print("Camera 1 (Reference):")
    print(pred_pose1)
    print("\nCamera 2:")
    print(pred_pose2)
    
    # Calculate predicted baseline from camera poses
    pred_t1 = pred_pose1[:3, 3]  # Translation vector of first camera
    pred_t2 = pred_pose2[:3, 3]  # Translation vector of second camera
    pred_baseline = np.linalg.norm(pred_t2.cpu().numpy() - pred_t1.cpu().numpy())
    print(f"Predicted Baseline: {pred_baseline}")
    
    return pred_depth, inference_time, pred_baseline, scene


def compare_and_visualize(img1, pred_depth, depth_gt, item_id, out_dir):
    """
    Creates a 2x2 plot comparing predicted depth with ground truth.
    Saves the plot to a file.
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    if isinstance(depth_gt, torch.Tensor):
        depth_gt = depth_gt.cpu().numpy()

    # Debug prints for depth values
    print("\nDebug depth values:")
    print(f"Pred depth shape: {pred_depth.shape}, dtype: {pred_depth.dtype}")
    print(f"Pred depth min: {np.min(pred_depth)}, max: {np.max(pred_depth)}")
    print(f"Pred depth has NaN: {np.isnan(pred_depth).any()}")
    print(f"Pred depth has inf: {np.isinf(pred_depth).any()}")
    print(f"GT depth shape: {depth_gt.shape}, dtype: {depth_gt.dtype}")
    print(f"GT depth min: {np.min(depth_gt)}, max: {np.max(depth_gt)}")
    print(f"GT depth has NaN: {np.isnan(depth_gt).any()}")
    print(f"GT depth has inf: {np.isinf(depth_gt).any()}")

    # Squeeze out channel dimension if it exists
    if depth_gt.ndim == 3 and depth_gt.shape[0] == 1:
        depth_gt = np.squeeze(depth_gt, axis=0)
    
    # Handle invalid values in predicted depth
    pred_depth = np.nan_to_num(pred_depth, nan=0.0, posinf=0.0, neginf=0.0)
    
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
    
    print(f"\nValid depth ranges:")
    print(f"Valid pred depth min: {np.min(valid_pred_depth)}, max: {np.max(valid_pred_depth)}")
    print(f"Valid GT depth min: {np.min(valid_depth_gt)}, max: {np.max(valid_depth_gt)}")
    
    vmin, vmax = None, None
    if valid_pred_depth.size > 0 and valid_depth_gt.size > 0:
        vmin = min(np.min(valid_pred_depth), np.min(valid_depth_gt))
        vmax = max(np.max(valid_pred_depth), np.max(valid_depth_gt))
        print(f"Using visualization range: vmin={vmin}, vmax={vmax}")
    else:
        logging.warning(f"Could not determine a valid color range for item {item_id}. Using separate color bars.")
        print("Warning: Could not determine valid color range")

    im = axes[1, 0].imshow(depth_gt, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Ground Truth Depth")
    axes[1, 0].axis('off')
    fig.colorbar(im, ax=axes[1, 0])
    
    im = axes[1, 1].imshow(pred_depth, cmap='viridis')#, vmin=vmin, vmax=vmax)
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
    output_path = os.path.join(out_dir, f"{item_id}_depth_comparison.png")
    plt.savefig(output_path)
    logging.info(f"Saved comparison plot to {output_path}")
    plt.close(fig)


def main(args):
    model = load_model(args)
    resource_manager = Boto3ResourceManager()

    # Initialize DataFrame to store results
    results_df = pd.DataFrame(columns=[
        'item_id', 'mean_error',
        'gt_min', 'gt_max', 'gt_mean',
        'pred_min', 'pred_max', 'pred_mean',
        'inference_time'
    ])

    def data_fn():
        df = pd.read_parquet(args.meta_data_path)
        for i in range(len(df)):
            idx = np.random.randint(0, len(df))
            yield DepthData.from_row(df.iloc[idx])

    processed_count = 0
    for data in data_fn():
        if args.limit_num is not None and processed_count >= args.limit_num:
            break
            
        logging.info(f"Processing item {data.item_id}")
        try:
            img1, img2, intrinsics, depth_gt, gt_baseline, gt_pose1, gt_pose2 = deserialize_data(
                data, resource_manager, args)
        except ValueError as e:
            logging.warning(f"Skipping item {data.item_id} due to: {e}")
            continue

        pred_depth_low_res, inference_time, _, scene = run_dust3r_inference(
            model, img1, img2, intrinsics, args, gt_pose1, gt_pose2)
        
        # Save 3D model as PLY
        print(f"Saving 3D model for item {data.item_id}...")
        try:
            model_filename = f"{data.item_id}_model.ply"
            model_output_path = get_3D_model_from_scene(
                outdir=args.out_dir,
                silent=False,
                scene=scene,
                glb_name=model_filename
            )
            if model_output_path:
                print(f"Saved 3D model to {model_output_path}")
            else:
                print(f"Warning: Could not generate or save 3D model for item {data.item_id}.")
        except Exception as e:
            print(f"Error saving 3D model for item {data.item_id}: {e}")
        
        # Resize predicted depth to match ground truth depth resolution
        H, W = img1.shape[2:]
        pred_depth_tensor = torch.from_numpy(pred_depth_low_res).unsqueeze(0).unsqueeze(0)
        pred_depth_resized = torch.nn.functional.interpolate(pred_depth_tensor, size=(H, W), mode='bilinear', align_corners=False)
        pred_depth = pred_depth_resized.squeeze().cpu().numpy()

        print(f"Pred depth min: {pred_depth.min()}, max: {pred_depth.max()}, mean: {pred_depth.mean()}")
        print(f"Pred depth shape: {pred_depth.shape}")
        
        # Record statistics
        stats = {
            'item_id': data.item_id,
            'mean_error': np.mean(np.abs(pred_depth - depth_gt.cpu().numpy())),
            'gt_min': depth_gt.min(),
            'gt_max': depth_gt.max(),
            'gt_mean': depth_gt.mean(),
            'pred_min': pred_depth.min(),
            'pred_max': pred_depth.max(),
            'pred_mean': pred_depth.mean(),
            'inference_time': inference_time
        }
        results_df = pd.concat([results_df, pd.DataFrame([stats])], ignore_index=True)
        
        print(f"\nDepth Statistics for item {data.item_id}:")
        print(f"Ground Truth - min: {depth_gt.min():.3f}, max: {depth_gt.max():.3f}, mean: {depth_gt.mean():.3f}")
        print(f"Predicted - min: {pred_depth.min():.3f}, max: {pred_depth.max():.3f}, mean: {pred_depth.mean():.3f}")
        print(f"Mean absolute error: {stats['mean_error']:.3f}")
        
        if isinstance(pred_depth, torch.Tensor):
            pred_depth = pred_depth.cpu().numpy()

        compare_and_visualize(img1, pred_depth, depth_gt, data.item_id, args.out_dir)

        logging.info(
            f"Inference time: {inference_time:.4f}s, Predicted depth map shape: {pred_depth.shape}")
        
        processed_count += 1
        
        # Save results every 10 scenes
        if processed_count % 10 == 0:
            results_path = os.path.join(args.out_dir, 'depth_benchmark_results.csv')
            results_df.to_csv(results_path, index=False)
            logging.info(f"Saved results to {results_path}")
    
    # Save final results
    results_path = os.path.join(args.out_dir, 'depth_benchmark_results.csv')
    results_df.to_csv(results_path, index=False)
    logging.info(f"Saved final results to {results_path}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total scenes processed: {processed_count}")
    print(f"Mean error: {results_df['mean_error'].mean():.3f} ± {results_df['mean_error'].std():.3f}")
    print(f"Mean inference time: {results_df['inference_time'].mean():.3f}s ± {results_df['inference_time'].std():.3f}s")


if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description="Run DUSt3R depth estimation and compare with ground truth.")
    
    # Model arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--weights", type=str, help="Path to DUSt3R model weights (.pth file).")
    model_group.add_argument("--model_name", type=str, default="DUSt3R_ViTLarge_BaseDecoder_512_dpt", help="Name of the model from HuggingFace Hub (e.g., 'DUSt3R_ViTLarge_BaseDecoder_512_dpt').")

    # Data arguments
    parser.add_argument('--meta_data_path', default="metadata/depth_live_1724981057", type=str, help='Path to metadata parquet file.')
    
    # Output arguments
    parser.add_argument('--out_dir', default=f'{code_dir}/../output/dust3r_benchmark/', type=str, help='The directory to save results.')

    # Inference arguments
    parser.add_argument("--device", type=str, default='cuda', help="PyTorch device to use ('cuda' or 'cpu').")
    parser.add_argument("--image_size", type=int, default=512, choices=[224, 512], help="Image size for DUSt3R processing. Default: 512.")
    parser.add_argument("--limit-num", type=int, help="Limit the number of items to process. If not set, process all items.")

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        args.device = 'cpu'

    print("Starting DUSt3R depth benchmark...")
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    main(args)
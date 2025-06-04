import argparse
import os
import torch
import numpy as np
import copy
import glob
import json
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# DUSt3R imports
from dust3r.model import load_model, AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# MODIFIED: main now takes model and a list of basenames
def main(model, args, basenames_list):

    if not basenames_list:
        print("No image pair basenames to process.")
        return

    for current_basename in basenames_list:
        print(f"\n--- Processing basename: {current_basename} ---")

        # Specific pair mode is now the only mode
        # base_name = current_basename # This was the old parameter, now using current_basename directly
        pair_dir = args.image_pair_dir
        left_path = os.path.join(pair_dir, f"{current_basename}_left.png")
        right_path = os.path.join(pair_dir, f"{current_basename}_right.png")

        print(f"Processing specific pair: \n  Left: {left_path}\n  Right: {right_path}")

        if not os.path.exists(left_path) or not os.path.exists(right_path):
            print(f"Error: One or both images for the pair not found. Searched for:\n  {left_path}\n  {right_path}")
            print("Please ensure both files exist and the paths are correct. Skipping this pair.")
            continue # Skip to the next basename

        # Model is already loaded and passed as an argument
        # os.makedirs(args.output_dir, exist_ok=True) # Output dir created once in __main__

        loaded_imgs_all = load_images([left_path, right_path], size=args.image_size, verbose=True)
        
        pairs = make_pairs(loaded_imgs_all, prefilter=None, symmetrize=True)

        output = inference(pairs, model, args.device, batch_size=1, verbose=True)

        print("Performing global alignment...")
        # For a single pair, PairViewer mode is appropriate.
        # If multiple pairs were processed for a single scene, PointCloudOptimizer might be used,
        # but here each pair is processed independently.
        mode = GlobalAlignerMode.PairViewer 
        scene = global_aligner(output, device=args.device, mode=mode, verbose=True)

        # Global alignment optimization is typically for >2 images.
        # Since we process pairs independently, full optimization per pair might be much.
        # The original logic for niter was conditional on len(loaded_imgs_all) > 2
        # which for a single pair is false. The demo.py uses PairViewer for 2 images.
        # If optimization for each pair is desired, it can be added here.
        # For now, sticking to PairViewer for individual pair processing.

        # Save camera parameters (intrinsics and poses)
        if scene.get_intrinsics() is not None and scene.get_im_poses() is not None:
            intrinsics_list = to_numpy(scene.get_intrinsics()).tolist()
            im_poses_list = to_numpy(scene.get_im_poses()).tolist()
            
            camera_params = {
                "intrinsics": intrinsics_list,
                "im_poses": im_poses_list
            }
            
            json_output_path = os.path.join(args.output_dir, f"{current_basename}_camera_parameters.json")
            try:
                with open(json_output_path, 'w') as f:
                    json.dump(camera_params, f, indent=4)
                print(f"Saved camera parameters to {json_output_path}")
            except Exception as e:
                print(f"Error saving camera parameters for {current_basename} to {json_output_path}: {e}")
        else:
            print(f"Warning: Could not retrieve intrinsics or poses for {current_basename}. Skipping camera parameter saving.")

        print("Saving RGB and Depth images...")
        rgb_images = scene.imgs
        depth_maps_tensor = scene.get_depthmaps()
        if depth_maps_tensor is None or len(depth_maps_tensor) == 0:
            print(f"Error: No depth maps found for basename {current_basename}. Cannot save depth images.")
            continue # Skip to the next basename
        depth_maps = to_numpy(depth_maps_tensor)

        if len(rgb_images) == 2 and len(depth_maps) == 2:
            left_rgb_path = os.path.join(args.output_dir, f"{current_basename}_left_rgb.png")
            right_rgb_path = os.path.join(args.output_dir, f"{current_basename}_right_rgb.png")
            left_depth_path = os.path.join(args.output_dir, f"{current_basename}_left_depth_colored.png")
            right_depth_path = os.path.join(args.output_dir, f"{current_basename}_right_depth_colored.png")

            plt.imsave(left_rgb_path, rgb_images[0])
            print(f"Saved left RGB image to {left_rgb_path}")
            plt.imsave(right_rgb_path, rgb_images[1])
            print(f"Saved right RGB image to {right_rgb_path}")

            # Save colored depth maps
            plt.imsave(left_depth_path, depth_maps[0], cmap='viridis')
            print(f"Saved left colored depth image to {left_depth_path}")
            plt.imsave(right_depth_path, depth_maps[1], cmap='viridis')
            print(f"Saved right colored depth image to {right_depth_path}")

            # Save raw depth maps
            left_depth_raw_path = os.path.join(args.output_dir, f"{current_basename}_left_depth_raw.png")
            right_depth_raw_path = os.path.join(args.output_dir, f"{current_basename}_right_depth_raw.png")
            
            plt.imsave(left_depth_raw_path, depth_maps[0], cmap='gray')
            print(f"Saved left raw depth image to {left_depth_raw_path}")
            plt.imsave(right_depth_raw_path, depth_maps[1], cmap='gray')
            print(f"Saved right raw depth image to {right_depth_raw_path}")
            
            print(f"Images saved successfully for {current_basename}.")
        else:
            print(f"Error: Expected 2 RGB images and 2 depth maps for {current_basename}, but found {len(rgb_images)} RGBs and {len(depth_maps)} depths.")
            # print("RGB images content:", rgb_images) # Potentially large output
            # print("Depth maps content:", depth_maps) # Potentially large output
        
        print(f"--- Finished processing basename: {current_basename} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process image pairs to generate RGB and colored depth map outputs using DUSt3R. \
                     Accepts one or more specific basenames via --input_pair_basename, \
                     or scans --image_pair_dir for all pairs if --input_pair_basename is omitted."
    )
    
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--weights", type=str, help="Path to the model weights (.pth file).")
    model_group.add_argument("--model_name", type=str, help="Name of the model (e.g., 'DUSt3R_ViTLarge_BaseDecoder_512_dpt') for HuggingFace Hub or local cache.")

    parser.add_argument("--input_pair_basename", type=str, nargs='+', default=None, # MODIFIED: nargs='+' for list, still optional
                        help="Optional. One or more basenames (e.g., 'image_001' 'image_002') of image pairs (_left.png/_right.png). \
                              If provided, only these pairs will be processed from the --image_pair_dir. \
                              If omitted, all pairs in --image_pair_dir will be scanned and processed.")
    parser.add_argument("--image_pair_dir", type=str, required=True, 
                        help="Directory containing the image pairs. If --input_pair_basename is given, this is where those pairs are located. \
                              If --input_pair_basename is omitted, this directory will be scanned for all pairs.")
    
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output images.")
    
    parser.add_argument("--device", type=str, default='cuda', help="PyTorch device to use ('cuda' or 'cpu'). Default: 'cuda'.")
    parser.add_argument("--image_size", type=int, default=512, choices=[224, 512], help="Image size for processing. Default: 512.")
    # niter argument is less relevant if PairViewer mode is always used for individual pairs. Kept for consistency.
    parser.add_argument("--niter", type=int, default=300, help="Number of iterations for global alignment (used by PointCloudOptimizer mode, less relevant for PairViewer mode per pair).")
    
    parsed_args = parser.parse_args()
    
    if parsed_args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        parsed_args.device = 'cpu'

    # Create output directory once
    os.makedirs(parsed_args.output_dir, exist_ok=True)

    # Load model once
    print(f"Loading model... Device: {parsed_args.device}")
    if parsed_args.weights:
        model = load_model(parsed_args.weights, parsed_args.device)
    else: # parsed_args.model_name must be set
        model = AsymmetricCroCo3DStereo.from_pretrained(parsed_args.model_name, device=parsed_args.device)
    print("Model loaded.")

    basenames_to_process = []
    if parsed_args.input_pair_basename:
        # If specific basenames are provided (it's now a list)
        basenames_to_process = parsed_args.input_pair_basename
        print(f"Processing specific basenames from arguments: {basenames_to_process}")
        # Ensure the image_pair_dir is valid
        if not os.path.isdir(parsed_args.image_pair_dir):
            print(f"Error: Image pair directory not found or is not a directory: {parsed_args.image_pair_dir}")
            exit(1)
    else:
        # If no specific basenames are provided, scan the directory for all pairs
        print(f"No specific input_pair_basename provided. Scanning directory: {parsed_args.image_pair_dir}")
        image_input_dir = parsed_args.image_pair_dir
        if not os.path.isdir(image_input_dir):
            print(f"Error: Image pair directory not found or is not a directory: {image_input_dir}")
            exit(1) 

        found_basenames_set = set()
        for filename in os.listdir(image_input_dir):
            if filename.endswith("_left.png"):
                basename = filename[:-9] # len("_left.png") == 9
                # Also check if corresponding _right.png exists
                right_file_path = os.path.join(image_input_dir, f"{basename}_right.png")
                if basename and os.path.exists(right_file_path):
                    found_basenames_set.add(basename)
            # No need to check _right.png separately if we ensure _left implies _right check
            # elif filename.endswith("_right.png"):
            #     basename = filename[:-10] # len("_right.png") == 10
            #     # Also check if corresponding _left.png exists
            #     left_file_path = os.path.join(image_input_dir, f"{basename}_left.png")
            #     if basename and os.path.exists(left_file_path):
            #         found_basenames_set.add(basename)
        
        if not found_basenames_set:
            print(f"No valid image pairs (e.g., xxx_left.png and xxx_right.png) found in {image_input_dir}")
            exit(1)

        basenames_to_process = sorted(list(found_basenames_set))
        print(f"Found {len(basenames_to_process)} unique image pair basenames in '{image_input_dir}': {basenames_to_process}")

    # Call main once with the list of basenames
    if basenames_to_process:
        main(model, parsed_args, basenames_to_process)
        print("\n--- All specified basenames processed. ---")
    else:
        print("No basenames were identified for processing.")

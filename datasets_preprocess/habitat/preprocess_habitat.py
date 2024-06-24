#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# main executable for preprocessing habitat
# export METADATA_DIR="/path/to/habitat/5views_v1_512x512_metadata"
# export SCENES_DIR="/path/to/habitat/data/scene_datasets/"
# export OUTPUT_DIR="data/habitat_processed"
# export PYTHONPATH=$(pwd)
# python preprocess_habitat.py --scenes_dir=$SCENES_DIR --metadata_dir=$METADATA_DIR --output_dir=$OUTPUT_DIR | parallel -j 16
# --------------------------------------------------------
import os
import glob
import json
import os

import PIL.Image
import json
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # noqa
import cv2
from habitat_renderer import multiview_crop_generator
from tqdm import tqdm


def preprocess_metadata(metadata_filename,
                        scenes_dir,
                        output_dir,
                        crop_resolution=[512, 512],
                        equirectangular_resolution=None,
                        fix_existing_dataset=False):
    # Load data
    with open(metadata_filename, "r") as f:
        metadata = json.load(f)

    if metadata["scene_dataset_config_file"] == "":
        scene = os.path.join(scenes_dir, metadata["scene"])
        scene_dataset_config_file = ""
    else:
        scene = metadata["scene"]
        scene_dataset_config_file = os.path.join(scenes_dir, metadata["scene_dataset_config_file"])
    navmesh = None

    # Use 4 times the crop size as resolution for rendering the environment map.
    max_res = max(crop_resolution)

    if equirectangular_resolution == None:
        # Use 4 times the crop size as resolution for rendering the environment map.
        max_res = max(crop_resolution)
        equirectangular_resolution = (4*max_res, 8*max_res)

    print("equirectangular_resolution:", equirectangular_resolution)

    if os.path.exists(output_dir) and not fix_existing_dataset:
        raise FileExistsError(output_dir)

    # Lazy initialization
    highres_dataset = None

    for batch_label, batch in tqdm(metadata["view_batches"].items()):
        for view_label, view_params in batch.items():

            assert view_params["size"] == crop_resolution
            label = f"{batch_label}_{view_label}"

            output_camera_params_filename = os.path.join(output_dir, f"{label}_camera_params.json")
            if fix_existing_dataset and os.path.isfile(output_camera_params_filename):
                # Skip generation if we are fixing a dataset and the corresponding output file already exists
                continue

            # Lazy initialization
            if highres_dataset is None:
                highres_dataset = multiview_crop_generator.HabitatMultiviewCrops(scene=scene,
                                                                                 navmesh=navmesh,
                                                                                 scene_dataset_config_file=scene_dataset_config_file,
                                                                                 equirectangular_resolution=equirectangular_resolution,
                                                                                 crop_resolution=crop_resolution,)
                os.makedirs(output_dir, exist_ok=bool(fix_existing_dataset))

            # Generate a higher resolution crop
            original_projection, position = multiview_crop_generator.dict_to_perspective_projection(view_params)
            # Render an envmap at the given position
            viewpoint_data = highres_dataset.render_viewpoint_data(position)

            projection = original_projection
            colormap, depthmap, pointmap, _ = highres_dataset.extract_cropped_camera(
                projection, viewpoint_data.colormap, viewpoint_data.distancemap, viewpoint_data.pointmap)

            camera_params = multiview_crop_generator.perspective_projection_to_dict(projection, position)

            # Color image
            PIL.Image.fromarray(colormap).save(os.path.join(output_dir, f"{label}.jpeg"))
            # Depth image
            cv2.imwrite(os.path.join(output_dir, f"{label}_depth.exr"),
                        depthmap, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
            # Camera parameters
            with open(output_camera_params_filename, "w") as f:
                json.dump(camera_params, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_dir", required=True)
    parser.add_argument("--scenes_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metadata_filename", default="")

    args = parser.parse_args()

    if args.metadata_filename == "":
        # Walk through the metadata dir to generate commandlines
        for filename in glob.iglob(os.path.join(args.metadata_dir, "**/metadata.json"), recursive=True):
            output_dir = os.path.join(args.output_dir, os.path.relpath(os.path.dirname(filename), args.metadata_dir))
            if not os.path.exists(output_dir):
                commandline = f"python {__file__} --metadata_filename={filename} --metadata_dir={args.metadata_dir} --scenes_dir={args.scenes_dir} --output_dir={output_dir}"
                print(commandline)
    else:
        preprocess_metadata(metadata_filename=args.metadata_filename,
                            scenes_dir=args.scenes_dir,
                            output_dir=args.output_dir)

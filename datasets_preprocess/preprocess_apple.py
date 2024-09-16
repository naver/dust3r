import os
import glob
import cv2
import numpy as np
import shutil

root_dir = r"D:\Projects\dust3r\data\0908"
source_path = '3D'

rgb_output_dir = os.path.join(root_dir, 'output', 'images')
depth_output_dir = os.path.join(root_dir, 'output', 'depth')
os.makedirs(rgb_output_dir, exist_ok=True)
os.makedirs(depth_output_dir, exist_ok=True)

image_idx = 0
time_stamps = os.listdir(os.path.join(root_dir, source_path))
for idx, time_stamp in enumerate(time_stamps):
    rgb_path = glob.glob(os.path.join(root_dir,  source_path, time_stamp, 'Color', '*.jpg'))[0]
    depth_path = glob.glob(os.path.join(root_dir, source_path, time_stamp, 'DepthImgToColorSensor', '*.png'))[0]

    print(rgb_path, depth_path)

    shutil.copy(src=rgb_path,
                dst=os.path.join(rgb_output_dir, 'color_{}.jpg'.format(image_idx)))
    shutil.copy(src=depth_path,
                dst=os.path.join(depth_output_dir, 'depth_{}.png'.format(image_idx)))

    image_idx += 1

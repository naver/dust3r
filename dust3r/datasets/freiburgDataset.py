import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import os.path as osp
from PIL import Image
import cv2

import glob

import sys
sys.path.append("/home/user/elwakeely1/dust3r")
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import load_images,preprocess_ir_rgb
from dust3r.datasets.base.base_stereo_view_dataset import view_name
from dust3r.viz import SceneViz, auto_cam_size
from dust3r.utils.image import rgb

from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True


class freiburgDataset(BaseStereoViewDataset):
    def __init__(self, *args, ROOT,method , **kwargs):
        self.ROOT = ROOT
        self.method = method 
        super().__init__(*args, **kwargs)
        self.scenes= []
        self.pairs =[]
        self.frames =[]
        self._load_data()
    def load_train(self):
        self.scene_files = sorted(glob.glob(osp.join(self.ROOT,self.split,self.method, "dataset_seq_*.npz")))
        for scene_id, scene_file in enumerate(self.scene_files):
            with np.load(scene_file, allow_pickle=True) as data:
                frames = dict(data)
                self.frames.append(frames)
                self.scenes.append(scene_file) 
                for i in range(len(frames) - 1): 
                    if f"{i+1}" in frames and f"{i}" in frames : 
                        fm1_id = frames[f"{i}"].item()["img_number"]
                        fm2_id = frames[f"{i+1}"].item()["img_number"]
                        if fm2_id == fm1_id + 1:
                            self.pairs.append((scene_file,scene_id, fm1_id, fm2_id))                
    
    def load_test(self):
        data = np.load(osp.join(self.ROOT,self.split, "dataset_test.npz"),allow_pickle=True)
        self.frames = dict(data)
        

    def _load_data(self):

        if self.split == "Test":
            self.load_test()
        if self.split =="Train":
            self.load_train()
    
    def get_view_train(self,pair_idx, resolution, rng):
        seq_path,seq, fm1, fm2 = self.pairs[pair_idx]
        seq_frames = self.frames[seq]
        views = []
        for view_index in [fm1, fm2]:
            data = seq_frames[f"{view_index}"].item()
            IR_img_path = data["IR_aligned_path"]
            ir_img = Image.open(str(IR_img_path))
            rgb_path = data["RGB_path"]
            rgb_image = Image.open(str(rgb_path))
            rgb,ir_img = preprocess_ir_rgb(rgb_image,ir_img)
            depthmap = data["Depth"]
            intrinsics =np.float32( data["Camera_intrinsic"])
            camera_pose = np.float32(data["camera_pose"])     
            views.append(dict(
                img=ir_img,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='freiburg',
                label=str(seq_path),
                instance=str(IR_img_path)))
        return views 
        
    def get_view_test(self,pair_idx, resolution, rng):
        views = []
        for i in range(2):
            data = self.frames[f"{pair_idx}"].item()
            IR_img_path = data["IR_aligned_path"]
            ir_img = Image.open(str(IR_img_path))
            rgb_path = data["RGB_path"]
            rgb_image = Image.open(str(rgb_path))
            rgb,ir_img = preprocess_ir_rgb(rgb_image,ir_img)
            depthmap = data["Depth"]
            intrinsics =np.float32( data["Camera_intrinsic"])
            views.append(dict(
                    img=ir_img,
                    depthmap=depthmap,
                    camera_intrinsics=intrinsics,
                    dataset='freiburg',
                    label=0,
                    instance=str(IR_img_path)))
        return views 


    def _get_views(self, pair_idx, resolution, rng):
        if self.split == "Test":
            views = self.get_view_test(pair_idx, resolution, rng)
        if self.split =="Train":
            views = self.get_view_train(pair_idx, resolution, rng)
        return views        

    def __len__(self):
        """Returns the number of samples in the dataset."""        
        if self.split == "Train":
            return len(self.pairs)
        elif self.split == "Test":
            return len(self.frames)

if __name__ == "__main__":

    
    train_ds = freiburgDataset(ROOT="/home/user/elwakeely1/DataParam",method="PNP", split = "Test",resolution=224, aug_crop=16)
    views = train_ds[0]
    print(views[0]["img"])
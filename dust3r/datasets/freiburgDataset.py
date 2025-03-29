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
from dust3r.utils.image import resize_img,preprocess_ir_rgb
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

    def _resize_pil_image(self,img, long_edge_size):
        S = max(img.size)  # Get the longest dimension
        if S > long_edge_size:
            interp = Image.LANCZOS  # High-quality downscaling
        else:
            interp = Image.BICUBIC  # Smooth upscaling

        new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
        return img.resize(new_size, interp)

    def crop_resize(self,img, size=224):
        W1, H1 = img.size

        if size == 224:
            img = self._resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))

        W, H = img.size  
        cx, cy = W // 2, H // 2  
        half = min(cx, cy)

        img = img.crop((cx - half, cy - half, cx + half, cy + half))
        return img
    
    def applyClaheToPIL(self, image):
        # Convert PIL image to NumPy array (OpenCV format)
        image_np = np.array(image)  
        
        # Convert RGB to LAB
        lab_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        
        # Split into channels and convert tuple to list so it becomes mutable
        lab_planes = list(cv2.split(lab_image))
        
        # Apply CLAHE to the L-channel (brightness)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        
        # Merge back and convert to RGB
        enhanced_image = cv2.merge(lab_planes)
        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2RGB)
        
        # Convert back to PIL image
        return Image.fromarray(enhanced_image)

    
    def get_view_train(self,pair_idx, resolution, rng):
        seq_path,seq, fm1, fm2 = self.pairs[pair_idx]
        seq_frames = self.frames[seq]
        views = []
        for view_index in [fm1, fm2]:
            data = seq_frames[f"{view_index}"].item()
            IR_img_path = data["IR_aligned_path"]
            # image = _crop([str(IR_img_path)],size = 224)
            image = Image.open(IR_img_path).convert("RGB")
            enhanced_image = self.applyClaheToPIL(image)
            image = self.crop_resize(image)
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
            # image = _crop([str(IR_img_path)],size = 224)
            image = Image.open(IR_img_path).convert("RGB")
            enhanced_image = self.applyClaheToPIL(image)
            image = self.crop_resize(enhanced_image)
            depthmap = data["Depth"]
            intrinsics =np.float32( data["Camera_intrinsic"])
            views.append(dict(
                    img=image,
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

    
    train_ds = freiburgDataset(ROOT="/home/user/elwakeely1/DataParam",method="RANSAC", split = "Train",resolution=224, aug_crop=16)
    views = train_ds[0]
    print(views[0])

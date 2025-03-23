import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import os.path as osp
from PIL import Image
import PIL

import glob

import sys
sys.path.append("/home/user/zafara1/dust3r")
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import load_images
from dust3r.datasets.base.base_stereo_view_dataset import view_name
from dust3r.viz import SceneViz, auto_cam_size
from dust3r.utils.image import rgb
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class freiburgDataset(BaseStereoViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.scenes= []
        self.pairs =[]
        self.frames =[]
        self._load_data()
        
    def _load_data(self):
        self.scene_files = sorted(glob.glob(osp.join(self.ROOT,self.split, "dataset_seq_*.npz")))
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
    def _resize_pil_image(self,img, long_edge_size):
        S = max(img.size)  # Get the longest dimension
        if S > long_edge_size:
            interp = PIL.Image.LANCZOS  # High-quality downscaling
        else:
            interp = PIL.Image.BICUBIC  # Smooth upscaling

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

    def _get_views(self, pair_idx, resolution, rng):
        seq_path,seq, fm1, fm2 = self.pairs[pair_idx]
        seq_frames = self.frames[seq]
        views = []
        for view_index in [fm1, fm2]:
            data = seq_frames[f"{view_index}"].item()
            IR_img_path = data["IR_aligned_path"]
            # image = _crop([str(IR_img_path)],size = 224)
            image = Image.open(IR_img_path).convert("RGB")
            image = self.crop_resize(image)
            depthmap = data["Depth"]
            # depthmap = Image.fromarray(depth_path)
            intrinsics =np.float32( data["Camera_intrinsic"])
            camera_pose = np.float32(data["camera_pose"])     
            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='freiburg',
                label=str(seq_path),
                instance=str(IR_img_path)))
        return views        

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.pairs)

if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    
    train_ds = freiburgDataset(ROOT="/home/user/zafara1/DataParam", split = "Train",resolution=224, aug_crop=16)
    views = train_ds[0]


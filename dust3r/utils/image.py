# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions about images (loading/converting...)
# --------------------------------------------------------
import os
import torch
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa
from PIL import Image 

try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def img_to_arr( img ):
    if isinstance(img, str):
        img = imread_cv2(img)
    return img

def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rgb(ftensor, true_shape=None):
    if isinstance(ftensor, list):
        return [rgb(x, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return img.clip(min=0, max=1)


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

def preprocess_ir_rgb(img_rgb,img_ir):

    # Convert to NumPy array
    arr = np.array(img_ir, dtype=np.float32)

    # Normalize to range 0-255
    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255  # Normalize between 0-255
    arr = arr.astype(np.uint8)  # Convert to 8-bit
    img_ir = Image.fromarray(arr)
    img_ir_array = np.array(img_ir)
    # Threshold to detect the dark borders 
    threshold = 0 
    cols_mean = img_ir_array.mean(axis=0)  # Compute column-wise mean
    valid_cols = np.where(cols_mean > threshold)[0]  # Find valid (non-dark) columns

    # Crop based on valid columns
    left, right = valid_cols[0], valid_cols[-1]
    img_ir_cropped = img_ir.crop((left, 0, right, img_ir.height))
    img_cropped = img_rgb.crop((left, 0, right, img_rgb.height))

    # Resize to match RGB image dimensions
    img_ir_resized = img_ir_cropped.resize(img_rgb.size)
    img_rgb = img_cropped.resize(img_rgb.size)

    # Apply CLAHE (Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    img_ir_array = np.array(img_ir_resized)
    img_ir = clahe.apply(img_ir_array )
    img_ir = Image.fromarray(img_ir).convert("RGB")
    
    return img_rgb,img_ir



def load_images_dust3r(images,i=0,train=True):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    imgs = []
    for img in images:
        if train:
            imgs.append({
                'img': ImgNorm(img)[None],  
                'true_shape': np.int32([img.size[::-1]]),  
                'idx': i,  
                'instance': str(i)  
            })
        else:
            imgs.append({
                'img': img.unsqueeze(0), 
                'true_shape': np.int32([[224,224]]), 
                'idx': i, 
                'instance': str(i)
            })

    return imgs 

def resize_img(img, size, square_ok=False):
    """ Open and convert all images in a list or folder to proper input format for DUSt3R """

    W1, H1 = img.size
    if size == 224:
        img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
    else:
        img = _resize_pil_image(img, size)
    
    W, H = img.size
    cx, cy = W//2, H//2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx-half, cy-half, cx+half, cy+half))
    else:
        halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
        if not square_ok and W == H:
            halfh = 3 * halfw / 4
        img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))


    return img  

def load_images(img_list, size, square_ok=False,train = True):
    """Open and convert all images in a list or folder to proper input format for DUSt3R"""
    
    # Ensure img_list is a list
    if not isinstance(img_list, list):
        img_list = [img_list]

    imgs = []
    
    for idx, img in enumerate(img_list):
        img = resize_img(img, size, square_ok)
        dust3r_images = load_images_dust3r([img],idx,train)  
        imgs.append(dust3r_images[0])  

    return imgs[0] if len(imgs) == 1 else imgs

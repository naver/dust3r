# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# croppping utilities
# --------------------------------------------------------
import PIL.Image
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa
import numpy as np  # noqa
from dust3r.utils.geometry import colmap_to_opencv_intrinsics, opencv_to_colmap_intrinsics  # noqa
try:
    lanczos = PIL.Image.Resampling.LANCZOS
except AttributeError:
    lanczos = PIL.Image.LANCZOS


class ImageList:
    """ Convenience class to aply the same operation to a whole set of images.
    """

    def __init__(self, images):
        if not isinstance(images, (tuple, list, set)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def to_pil(self):
        return tuple(self.images) if len(self.images) > 1 else self.images[0]

    @property
    def size(self):
        sizes = [im.size for im in self.images]
        assert all(sizes[0] == s for s in sizes)
        return sizes[0]

    def resize(self, *args, **kwargs):
        return ImageList(self._dispatch('resize', *args, **kwargs))

    def crop(self, *args, **kwargs):
        return ImageList(self._dispatch('crop', *args, **kwargs))

    def _dispatch(self, func, *args, **kwargs):
        return [getattr(im, func)(*args, **kwargs) for im in self.images]


def rescale_image_depthmap(image, depthmap, camera_intrinsics, output_resolution):
    """ Jointly rescale a (image, depthmap) 
        so that (out_width, out_height) >= output_res
    """
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W,H)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        # can also use this with masks instead of depthmaps
        assert tuple(depthmap.shape[:2]) == image.size[::-1]
    assert output_resolution.shape == (2,)
    # define output resolution
    scale_final = max(output_resolution / image.size) + 1e-8
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    # first rescale the image so that it contains the crop
    image = image.resize(output_resolution, resample=lanczos)
    if depthmap is not None:
        depthmap = cv2.resize(depthmap, output_resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)

    # no offset here; simple rescaling
    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics, input_resolution, output_resolution, scaling=scale_final)

    return image.to_pil(), depthmap, camera_intrinsics


def camera_matrix_of_crop(input_camera_matrix, input_resolution, output_resolution, scaling=1, offset_factor=0.5, offset=None):
    # Margins to offset the origin
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins

    # Generate new camera parameters
    output_camera_matrix_colmap = opencv_to_colmap_intrinsics(input_camera_matrix)
    output_camera_matrix_colmap[:2, :] *= scaling
    output_camera_matrix_colmap[:2, 2] -= offset
    output_camera_matrix = colmap_to_opencv_intrinsics(output_camera_matrix_colmap)

    return output_camera_matrix


def crop_image_depthmap(image, depthmap, camera_intrinsics, crop_bbox):
    """
    Return a crop of the input view.
    """
    image = ImageList(image)
    l, t, r, b = crop_bbox

    image = image.crop((l, t, r, b))
    depthmap = depthmap[t:b, l:r]

    camera_intrinsics = camera_intrinsics.copy()
    camera_intrinsics[0, 2] -= l
    camera_intrinsics[1, 2] -= t

    return image.to_pil(), depthmap, camera_intrinsics


def bbox_from_intrinsics_in_out(input_camera_matrix, output_camera_matrix, output_resolution):
    out_width, out_height = output_resolution
    l, t = np.int32(np.round(input_camera_matrix[:2, 2] - output_camera_matrix[:2, 2]))
    crop_bbox = (l, t, l+out_width, t+out_height)
    return crop_bbox

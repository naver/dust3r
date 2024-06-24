# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Various 3D/2D projection utils, useful to sample virtual cameras.
# --------------------------------------------------------
import numpy as np

class EquirectangularProjection:
    """
    Convention for the central pixel of the equirectangular map similar to OpenCV perspective model:
        +X from left to right
        +Y from top to bottom
        +Z going outside the camera
    EXCEPT that the top left corner of the image is assumed to have (0,0) coordinates (OpenCV assumes (-0.5,-0.5))
    """

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.u_scaling = (2 * np.pi) / self.width
        self.v_scaling = np.pi / self.height

    def unproject(self, u, v):
        """
        Args:
            u, v: 2D coordinates
        Returns:
            unnormalized 3D rays.
        """
        longitude = self.u_scaling * u - np.pi
        minus_latitude = self.v_scaling * v - np.pi/2

        cos_latitude = np.cos(minus_latitude)
        x, z = np.sin(longitude) * cos_latitude, np.cos(longitude) * cos_latitude
        y = np.sin(minus_latitude)

        rays = np.stack([x, y, z], axis=-1)
        return rays

    def project(self, rays):
        """
        Args:
            rays: Bx3 array of 3D rays.
        Returns:
            u, v: tuple of 2D coordinates.
        """
        rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
        x, y, z = [rays[..., i] for i in range(3)]

        longitude = np.arctan2(x, z)
        minus_latitude = np.arcsin(y)

        u = (longitude + np.pi) * (1.0 / self.u_scaling)
        v = (minus_latitude + np.pi/2) * (1.0 / self.v_scaling)
        return u, v


class PerspectiveProjection:
    """
    OpenCV convention:
    World space:
        +X from left to right
        +Y from top to bottom
        +Z going outside the camera
    Pixel space:
        +u from left to right
        +v from top to bottom
    EXCEPT that the top left corner of the image is assumed to have (0,0) coordinates (OpenCV assumes (-0.5,-0.5)).
    """

    def __init__(self, K, height, width):
        self.height = height
        self.width = width
        self.K = K
        self.Kinv = np.linalg.inv(K)

    def project(self, rays):
        uv_homogeneous = np.einsum("ik, ...k -> ...i", self.K, rays)
        uv = uv_homogeneous[..., :2] / uv_homogeneous[..., 2, None]
        return uv[..., 0], uv[..., 1]

    def unproject(self, u, v):
        uv_homogeneous = np.stack((u, v, np.ones_like(u)), axis=-1)
        rays = np.einsum("ik, ...k -> ...i", self.Kinv, uv_homogeneous)
        return rays


class RotatedProjection:
    def __init__(self, base_projection, R_to_base_projection):
        self.base_projection = base_projection
        self.R_to_base_projection = R_to_base_projection

    @property
    def width(self):
        return self.base_projection.width

    @property
    def height(self):
        return self.base_projection.height

    def project(self, rays):
        if self.R_to_base_projection is not None:
            rays = np.einsum("ik, ...k -> ...i", self.R_to_base_projection, rays)
        return self.base_projection.project(rays)

    def unproject(self, u, v):
        rays = self.base_projection.unproject(u, v)
        if self.R_to_base_projection is not None:
            rays = np.einsum("ik, ...k -> ...i", self.R_to_base_projection.T, rays)
        return rays

def get_projection_rays(projection, noise_level=0):
    """
    Return a 2D map of 3D rays corresponding to the projection.
    If noise_level > 0, add some jittering noise to these rays.
    """
    grid_u, grid_v = np.meshgrid(0.5 + np.arange(projection.width), 0.5 + np.arange(projection.height))
    if noise_level > 0:
        grid_u += np.clip(0, noise_level * np.random.uniform(-0.5, 0.5, size=grid_u.shape), projection.width)
        grid_v += np.clip(0, noise_level * np.random.uniform(-0.5, 0.5, size=grid_v.shape), projection.height)
    return projection.unproject(grid_u, grid_v)

def compute_camera_intrinsics(height, width, hfov):
    f = width/2 / np.tan(hfov/2 * np.pi/180)
    cu, cv = width/2, height/2
    return f, cu, cv

def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K

def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K
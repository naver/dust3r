# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Generate pairs of crops from a dataset of environment maps.
# --------------------------------------------------------
import os
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # noqa
import cv2
import collections
from habitat_renderer import projections, projections_conversions
from habitat_renderer.habitat_sim_envmaps_renderer import HabitatEnvironmentMapRenderer

ViewpointData = collections.namedtuple("ViewpointData", ["colormap", "distancemap", "pointmap", "position"])

class HabitatMultiviewCrops:
    def __init__(self,
                 scene,
                 navmesh,
                 scene_dataset_config_file,
                 equirectangular_resolution=(400, 800),
                 crop_resolution=(240, 320),
                 pixel_jittering_iterations=5,
                 jittering_noise_level=1.0):
        self.crop_resolution = crop_resolution

        self.pixel_jittering_iterations = pixel_jittering_iterations
        self.jittering_noise_level = jittering_noise_level

        # Instanciate the low resolution habitat sim renderer
        self.lowres_envmap_renderer = HabitatEnvironmentMapRenderer(scene=scene,
                                                                    navmesh=navmesh,
                                                                    scene_dataset_config_file=scene_dataset_config_file,
                                                                    equirectangular_resolution=equirectangular_resolution,
                                                                    render_depth=True,
                                                                    render_equirectangular=True)
        self.R_cam_to_world = np.asarray(self.lowres_envmap_renderer.R_cam_to_world())
        self.up_direction = np.asarray(self.lowres_envmap_renderer.up_direction())

        # Projection applied by each environment map
        self.envmap_height, self.envmap_width = self.lowres_envmap_renderer.equirectangular_resolution
        base_projection = projections.EquirectangularProjection(self.envmap_height, self.envmap_width)
        self.envmap_projection = projections.RotatedProjection(base_projection, self.R_cam_to_world.T)
        # 3D Rays map associated to each envmap
        self.envmap_rays = projections.get_projection_rays(self.envmap_projection)

    def compute_pointmap(self, distancemap, position):
        # Point cloud associated to each ray
        return self.envmap_rays * distancemap[:, :, None] + position

    def render_viewpoint_data(self, position):
        data = self.lowres_envmap_renderer.render_viewpoint(np.asarray(position))
        colormap = data['observations']['color_equirectangular'][..., :3]  # Ignore the alpha channel
        distancemap = data['observations']['depth_equirectangular']
        pointmap = self.compute_pointmap(distancemap, position)
        return ViewpointData(colormap=colormap, distancemap=distancemap, pointmap=pointmap, position=position)

    def extract_cropped_camera(self, projection, color_image, distancemap, pointmap, voxelmap=None):
        remapper = projections_conversions.RemapProjection(input_projection=self.envmap_projection, output_projection=projection,
                                                           pixel_jittering_iterations=self.pixel_jittering_iterations, jittering_noise_level=self.jittering_noise_level)
        cropped_color_image = remapper.convert(
            color_image, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP, single_map=False)
        cropped_distancemap = remapper.convert(
            distancemap, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP, single_map=True)
        cropped_pointmap = remapper.convert(pointmap, interpolation=cv2.INTER_NEAREST,
                                            borderMode=cv2.BORDER_WRAP, single_map=True)
        cropped_voxelmap = (None if voxelmap is None else
                            remapper.convert(voxelmap, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP, single_map=True))
        # Convert the distance map into a depth map
        cropped_depthmap = np.asarray(
            cropped_distancemap / np.linalg.norm(remapper.output_rays, axis=-1), dtype=cropped_distancemap.dtype)

        return cropped_color_image, cropped_depthmap, cropped_pointmap, cropped_voxelmap

def perspective_projection_to_dict(persp_projection, position):
    """
    Serialization-like function."""
    camera_params = dict(camera_intrinsics=projections.colmap_to_opencv_intrinsics(persp_projection.base_projection.K).tolist(),
                         size=(persp_projection.base_projection.width, persp_projection.base_projection.height),
                         R_cam2world=persp_projection.R_to_base_projection.T.tolist(),
                         t_cam2world=position)
    return camera_params


def dict_to_perspective_projection(camera_params):
    K = projections.opencv_to_colmap_intrinsics(np.asarray(camera_params["camera_intrinsics"]))
    size = camera_params["size"]
    R_cam2world = np.asarray(camera_params["R_cam2world"])
    projection = projections.PerspectiveProjection(K, height=size[1], width=size[0])
    projection = projections.RotatedProjection(projection, R_to_base_projection=R_cam2world.T)
    position = camera_params["t_cam2world"]
    return projection, position
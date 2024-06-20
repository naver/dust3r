# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Remap data from one projection to an other
# --------------------------------------------------------
import numpy as np
import cv2
from habitat_renderer import projections

class RemapProjection:
    def __init__(self, input_projection, output_projection, pixel_jittering_iterations=0, jittering_noise_level=0):
        """
        Some naive random jittering can be introduced in the remapping to mitigate aliasing artecfacts.
        """
        assert jittering_noise_level >= 0
        assert pixel_jittering_iterations >= 0

        maps = []
        # Initial map
        self.output_rays = projections.get_projection_rays(output_projection)
        map_u, map_v = input_projection.project(self.output_rays)
        map_u, map_v = np.asarray(map_u, dtype=np.float32), np.asarray(map_v, dtype=np.float32)
        maps.append((map_u, map_v))

        for _ in range(pixel_jittering_iterations):
            # Define multiple mappings using some coordinates jittering to mitigate aliasing effects
            crop_rays = projections.get_projection_rays(output_projection, jittering_noise_level)
            map_u, map_v = input_projection.project(crop_rays)
            map_u, map_v = np.asarray(map_u, dtype=np.float32), np.asarray(map_v, dtype=np.float32)
            maps.append((map_u, map_v))
        self.maps = maps

    def convert(self, img, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP, single_map=False):
        remapped = []
        for map_u, map_v in self.maps:
            res = cv2.remap(img, map_u, map_v, interpolation=interpolation, borderMode=borderMode)
            remapped.append(res)
            if single_map:
                break
        if len(remapped) == 1:
            res = remapped[0]
        else:
            res = np.asarray(np.mean(remapped, axis=0), dtype=img.dtype)
        return res

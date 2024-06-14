# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# 7 Scenes dataloader
# --------------------------------------------------------
import os
import numpy as np
import torch
import PIL.Image

import kapture
from kapture.io.csv import kapture_from_dir
from kapture_localization.utils.pairsfile import get_ordered_pairs_from_file
from kapture.io.records import depth_map_from_file

from dust3r_visloc.datasets.utils import cam_to_world_from_kapture, get_resize_function, rescale_points3d
from dust3r_visloc.datasets.base_dataset import BaseVislocDataset
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates, xy_grid, geotrf


class VislocSevenScenes(BaseVislocDataset):
    def __init__(self, root, subscene, pairsfile, topk=1):
        super().__init__()
        self.root = root
        self.subscene = subscene
        self.topk = topk
        self.num_views = self.topk + 1
        self.maxdim = None
        self.patch_size = None

        query_path = os.path.join(self.root, subscene, 'query')
        kdata_query = kapture_from_dir(query_path)
        assert kdata_query.records_camera is not None and kdata_query.trajectories is not None and kdata_query.rigs is not None
        kapture.rigs_remove_inplace(kdata_query.trajectories, kdata_query.rigs)
        kdata_query_searchindex = {kdata_query.records_camera[(timestamp, sensor_id)]: (timestamp, sensor_id)
                                   for timestamp, sensor_id in kdata_query.records_camera.key_pairs()}
        self.query_data = {'path': query_path, 'kdata': kdata_query, 'searchindex': kdata_query_searchindex}

        map_path = os.path.join(self.root, subscene, 'mapping')
        kdata_map = kapture_from_dir(map_path)
        assert kdata_map.records_camera is not None and kdata_map.trajectories is not None and kdata_map.rigs is not None
        kapture.rigs_remove_inplace(kdata_map.trajectories, kdata_map.rigs)
        kdata_map_searchindex = {kdata_map.records_camera[(timestamp, sensor_id)]: (timestamp, sensor_id)
                                 for timestamp, sensor_id in kdata_map.records_camera.key_pairs()}
        self.map_data = {'path': map_path, 'kdata': kdata_map, 'searchindex': kdata_map_searchindex}

        self.pairs = get_ordered_pairs_from_file(os.path.join(self.root, subscene,
                                                              'pairfiles/query',
                                                              pairsfile + '.txt'))
        self.scenes = kdata_query.records_camera.data_list()

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        assert self.maxdim is not None and self.patch_size is not None
        query_image = self.scenes[idx]
        map_images = [p[0] for p in self.pairs[query_image][:self.topk]]
        views = []
        dataarray = [(query_image, self.query_data, False)] + [(map_image, self.map_data, True)
                                                               for map_image in map_images]
        for idx, (imgname, data, should_load_depth) in enumerate(dataarray):
            imgpath, kdata, searchindex = map(data.get, ['path', 'kdata', 'searchindex'])

            timestamp, camera_id = searchindex[imgname]

            # for 7scenes, SIMPLE_PINHOLE
            camera_params = kdata.sensors[camera_id].camera_params
            W, H, f, cx, cy = camera_params
            distortion = [0, 0, 0, 0]
            intrinsics = np.float32([(f, 0, cx),
                                     (0, f, cy),
                                     (0, 0, 1)])

            cam_to_world = cam_to_world_from_kapture(kdata, timestamp, camera_id)

            # Load RGB image
            rgb_image = PIL.Image.open(os.path.join(imgpath, 'sensors/records_data', imgname)).convert('RGB')
            rgb_image.load()

            W, H = rgb_image.size
            resize_func, to_resize, to_orig = get_resize_function(self.maxdim, self.patch_size, H, W)

            rgb_tensor = resize_func(ImgNorm(rgb_image))

            view = {
                'intrinsics': intrinsics,
                'distortion': distortion,
                'cam_to_world': cam_to_world,
                'rgb': rgb_image,
                'rgb_rescaled': rgb_tensor,
                'to_orig': to_orig,
                'idx': idx,
                'image_name': imgname
            }

            # Load depthmap
            if should_load_depth:
                depthmap_filename = os.path.join(imgpath, 'sensors/records_data',
                                                 imgname.replace('color.png', 'depth.reg'))
                depthmap = depth_map_from_file(depthmap_filename, (int(W), int(H))).astype(np.float32)
                pts3d_full, pts3d_valid = depthmap_to_absolute_camera_coordinates(depthmap, intrinsics, cam_to_world)

                pts3d = pts3d_full[pts3d_valid]
                pts2d_int = xy_grid(W, H)[pts3d_valid]
                pts2d = pts2d_int.astype(np.float64)

                # nan => invalid
                pts3d_full[~pts3d_valid] = np.nan
                pts3d_full = torch.from_numpy(pts3d_full)
                view['pts3d'] = pts3d_full
                view["valid"] = pts3d_full.sum(dim=-1).isfinite()

                HR, WR = rgb_tensor.shape[1:]
                _, _, pts3d_rescaled, valid_rescaled = rescale_points3d(pts2d, pts3d, to_resize, HR, WR)
                pts3d_rescaled = torch.from_numpy(pts3d_rescaled)
                valid_rescaled = torch.from_numpy(valid_rescaled)
                view['pts3d_rescaled'] = pts3d_rescaled
                view["valid_rescaled"] = valid_rescaled
            views.append(view)
        return views

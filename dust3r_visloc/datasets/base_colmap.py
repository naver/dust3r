# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Base class for colmap / kapture
# --------------------------------------------------------
import os
import numpy as np
from tqdm import tqdm
import collections
import pickle
import PIL.Image
import torch
from scipy.spatial.transform import Rotation
import torchvision.transforms as tvf

from kapture.core import CameraType
from kapture.io.csv import kapture_from_dir
from kapture_localization.utils.pairsfile import get_ordered_pairs_from_file

from dust3r_visloc.datasets.utils import cam_to_world_from_kapture, get_resize_function, rescale_points3d
from dust3r_visloc.datasets.base_dataset import BaseVislocDataset
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.utils.geometry import colmap_to_opencv_intrinsics

KaptureSensor = collections.namedtuple('Sensor', 'sensor_params camera_params')


def kapture_to_opencv_intrinsics(sensor):
    """
    Convert from Kapture to OpenCV parameters.
    Warning: we assume that the camera and pixel coordinates follow Colmap conventions here.
    Args:
        sensor: Kapture sensor
    """
    sensor_type = sensor.sensor_params[0]
    if sensor_type == "SIMPLE_PINHOLE":
        # Simple pinhole model.
        # We still call OpenCV undistorsion however for code simplicity.
        w, h, f, cx, cy = sensor.camera_params
        k1 = 0
        k2 = 0
        p1 = 0
        p2 = 0
        fx = fy = f
    elif sensor_type == "PINHOLE":
        w, h, fx, fy, cx, cy = sensor.camera_params
        k1 = 0
        k2 = 0
        p1 = 0
        p2 = 0
    elif sensor_type == "SIMPLE_RADIAL":
        w, h, f, cx, cy, k1 = sensor.camera_params
        k2 = 0
        p1 = 0
        p2 = 0
        fx = fy = f
    elif sensor_type == "RADIAL":
        w, h, f, cx, cy, k1, k2 = sensor.camera_params
        p1 = 0
        p2 = 0
        fx = fy = f
    elif sensor_type == "OPENCV":
        w, h, fx, fy, cx, cy, k1, k2, p1, p2 = sensor.camera_params
    else:
        raise NotImplementedError(f"Sensor type {sensor_type} is not supported yet.")

    cameraMatrix = np.asarray([[fx, 0, cx],
                               [0, fy, cy],
                               [0, 0, 1]], dtype=np.float32)

    # We assume that Kapture data comes from Colmap: the origin is different.
    cameraMatrix = colmap_to_opencv_intrinsics(cameraMatrix)

    distCoeffs = np.asarray([k1, k2, p1, p2], dtype=np.float32)
    return cameraMatrix, distCoeffs, (w, h)


def K_from_colmap(elems):
    sensor = KaptureSensor(elems, tuple(map(float, elems[1:])))
    cameraMatrix, distCoeffs, (w, h) = kapture_to_opencv_intrinsics(sensor)
    res = dict(resolution=(w, h),
               intrinsics=cameraMatrix,
               distortion=distCoeffs)
    return res


def pose_from_qwxyz_txyz(elems):
    qw, qx, qy, qz, tx, ty, tz = map(float, elems)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
    pose[:3, 3] = (tx, ty, tz)
    return np.linalg.inv(pose)  # returns cam2world


class BaseVislocColmapDataset(BaseVislocDataset):
    def __init__(self, image_path, map_path, query_path, pairsfile_path, topk=1, cache_sfm=False):
        super().__init__()
        self.topk = topk
        self.num_views = self.topk + 1
        self.image_path = image_path
        self.cache_sfm = cache_sfm

        self._load_sfm(map_path)

        kdata_query = kapture_from_dir(query_path)
        assert kdata_query.records_camera is not None and kdata_query.trajectories is not None

        kdata_query_searchindex = {kdata_query.records_camera[(timestamp, sensor_id)]: (timestamp, sensor_id)
                                   for timestamp, sensor_id in kdata_query.records_camera.key_pairs()}
        self.query_data = {'kdata': kdata_query, 'searchindex': kdata_query_searchindex}

        self.pairs = get_ordered_pairs_from_file(pairsfile_path)
        self.scenes = kdata_query.records_camera.data_list()

    def _load_sfm(self, sfm_dir):
        sfm_cache_path = os.path.join(sfm_dir, 'dust3r_cache.pkl')
        if os.path.isfile(sfm_cache_path) and self.cache_sfm:
            with open(sfm_cache_path, "rb") as f:
                data = pickle.load(f)
                self.img_infos = data['img_infos']
                self.points3D = data['points3D']
            return

        # load cameras
        with open(os.path.join(sfm_dir, 'cameras.txt'), 'r') as f:
            raw = f.read().splitlines()[3:]  # skip header

        intrinsics = {}
        for camera in tqdm(raw):
            camera = camera.split(' ')
            intrinsics[int(camera[0])] = K_from_colmap(camera[1:])

        # load images
        with open(os.path.join(sfm_dir, 'images.txt'), 'r') as f:
            raw = f.read().splitlines()
            raw = [line for line in raw if not line.startswith('#')]  # skip header

        self.img_infos = {}
        for image, points in tqdm(zip(raw[0::2], raw[1::2]), total=len(raw) // 2):
            image = image.split(' ')
            points = points.split(' ')

            img_name = image[-1]
            current_points2D = {int(i): (float(x), float(y))
                                for i, x, y in zip(points[2::3], points[0::3], points[1::3]) if i != '-1'}
            self.img_infos[img_name] = dict(intrinsics[int(image[-2])],
                                            path=img_name,
                                            camera_pose=pose_from_qwxyz_txyz(image[1: -2]),
                                            sparse_pts2d=current_points2D)

        # load 3D points
        with open(os.path.join(sfm_dir, 'points3D.txt'), 'r') as f:
            raw = f.read().splitlines()
            raw = [line for line in raw if not line.startswith('#')]  # skip header

        self.points3D = {}
        for point in tqdm(raw):
            point = point.split()
            self.points3D[int(point[0])] = tuple(map(float, point[1:4]))

        if self.cache_sfm:
            to_save = \
                {
                    'img_infos': self.img_infos,
                    'points3D': self.points3D
                }
            with open(sfm_cache_path, "wb") as f:
                pickle.dump(to_save, f)

    def __len__(self):
        return len(self.scenes)

    def _get_view_query(self, imgname):
        kdata, searchindex = map(self.query_data.get, ['kdata', 'searchindex'])

        timestamp, camera_id = searchindex[imgname]

        camera_params = kdata.sensors[camera_id].camera_params
        if kdata.sensors[camera_id].camera_type == CameraType.SIMPLE_PINHOLE:
            W, H, f, cx, cy = camera_params
            k1 = 0
            fx = fy = f
        elif kdata.sensors[camera_id].camera_type == CameraType.SIMPLE_RADIAL:
            W, H, f, cx, cy, k1 = camera_params
            fx = fy = f
        else:
            raise NotImplementedError('not implemented')

        W, H = int(W), int(H)
        intrinsics = np.float32([(fx, 0, cx),
                                 (0, fy, cy),
                                 (0, 0, 1)])
        intrinsics = colmap_to_opencv_intrinsics(intrinsics)
        distortion = [k1, 0, 0, 0]

        if kdata.trajectories is not None and (timestamp, camera_id) in kdata.trajectories:
            cam_to_world = cam_to_world_from_kapture(kdata, timestamp, camera_id)
        else:
            cam_to_world = np.eye(4, dtype=np.float32)

        # Load RGB image
        rgb_image = PIL.Image.open(os.path.join(self.image_path, imgname)).convert('RGB')
        rgb_image.load()
        resize_func, _, to_orig = get_resize_function(self.maxdim, self.patch_size, H, W)
        rgb_tensor = resize_func(ImgNorm(rgb_image))

        view = {
            'intrinsics': intrinsics,
            'distortion': distortion,
            'cam_to_world': cam_to_world,
            'rgb': rgb_image,
            'rgb_rescaled': rgb_tensor,
            'to_orig': to_orig,
            'idx': 0,
            'image_name': imgname
        }
        return view

    def _get_view_map(self, imgname, idx):
        infos = self.img_infos[imgname]

        rgb_image = PIL.Image.open(os.path.join(self.image_path, infos['path'])).convert('RGB')
        rgb_image.load()
        W, H = rgb_image.size
        intrinsics = infos['intrinsics']
        intrinsics = colmap_to_opencv_intrinsics(intrinsics)
        distortion_coefs = infos['distortion']

        pts2d = infos['sparse_pts2d']
        sparse_pos2d = np.float32(list(pts2d.values())).reshape((-1, 2))  # pts2d from colmap
        sparse_pts3d = np.float32([self.points3D[i] for i in pts2d]).reshape((-1, 3))

        # store full resolution 2D->3D
        sparse_pos2d_cv2 = sparse_pos2d.copy()
        sparse_pos2d_cv2[:, 0] -= 0.5
        sparse_pos2d_cv2[:, 1] -= 0.5
        sparse_pos2d_int = sparse_pos2d_cv2.round().astype(np.int64)
        valid = (sparse_pos2d_int[:, 0] >= 0) & (sparse_pos2d_int[:, 0] < W) & (
            sparse_pos2d_int[:, 1] >= 0) & (sparse_pos2d_int[:, 1] < H)
        sparse_pos2d_int = sparse_pos2d_int[valid]
        # nan => invalid
        pts3d = np.full((H, W, 3), np.nan, dtype=np.float32)
        pts3d[sparse_pos2d_int[:, 1], sparse_pos2d_int[:, 0]] = sparse_pts3d[valid]
        pts3d = torch.from_numpy(pts3d)

        cam_to_world = infos['camera_pose']  # cam2world

        # also store resized resolution 2D->3D
        resize_func, to_resize, to_orig = get_resize_function(self.maxdim, self.patch_size, H, W)
        rgb_tensor = resize_func(ImgNorm(rgb_image))

        HR, WR = rgb_tensor.shape[1:]
        _, _, pts3d_rescaled, valid_rescaled = rescale_points3d(sparse_pos2d_cv2, sparse_pts3d, to_resize, HR, WR)
        pts3d_rescaled = torch.from_numpy(pts3d_rescaled)
        valid_rescaled = torch.from_numpy(valid_rescaled)

        view = {
            'intrinsics': intrinsics,
            'distortion': distortion_coefs,
            'cam_to_world': cam_to_world,
            'rgb': rgb_image,
            "pts3d": pts3d,
            "valid": pts3d.sum(dim=-1).isfinite(),
            'rgb_rescaled': rgb_tensor,
            "pts3d_rescaled": pts3d_rescaled,
            "valid_rescaled": valid_rescaled,
            'to_orig': to_orig,
            'idx': idx,
            'image_name': imgname
        }
        return view

    def __getitem__(self, idx):
        assert self.maxdim is not None and self.patch_size is not None
        query_image = self.scenes[idx]
        map_images = [p[0] for p in self.pairs[query_image][:self.topk]]
        views = []
        views.append(self._get_view_query(query_image))
        for idx, map_image in enumerate(map_images):
            views.append(self._get_view_map(map_image, idx + 1))
        return views

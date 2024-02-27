# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Visualization utilities using trimesh
# --------------------------------------------------------
import PIL.Image
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from dust3r.utils.geometry import geotrf, get_med_dist_between_poses
from dust3r.utils.device import to_numpy
from dust3r.utils.image import rgb

try:
    import trimesh
except ImportError:
    print('/!\\ module trimesh is not installed, cannot visualize results /!\\')


def cat_3d(vecs):
    if isinstance(vecs, (np.ndarray, torch.Tensor)):
        vecs = [vecs]
    return np.concatenate([p.reshape(-1, 3) for p in to_numpy(vecs)])


def show_raw_pointcloud(pts3d, colors, point_size=2):
    scene = trimesh.Scene()

    pct = trimesh.PointCloud(cat_3d(pts3d), colors=cat_3d(colors))
    scene.add_geometry(pct)

    scene.show(line_settings={'point_size': point_size})


def pts3d_to_trimesh(img, pts3d, valid=None):
    H, W, THREE = img.shape
    assert THREE == 3
    assert img.shape == pts3d.shape

    vertices = pts3d.reshape(-1, 3)

    # make squares: each pixel == 2 triangles
    idx = np.arange(len(vertices)).reshape(H, W)
    idx1 = idx[:-1, :-1].ravel()  # top-left corner
    idx2 = idx[:-1, +1:].ravel()  # right-left corner
    idx3 = idx[+1:, :-1].ravel()  # bottom-left corner
    idx4 = idx[+1:, +1:].ravel()  # bottom-right corner
    faces = np.concatenate((
        np.c_[idx1, idx2, idx3],
        np.c_[idx3, idx2, idx1],  # same triangle, but backward (cheap solution to cancel face culling)
        np.c_[idx2, idx3, idx4],
        np.c_[idx4, idx3, idx2],  # same triangle, but backward (cheap solution to cancel face culling)
    ), axis=0)

    # prepare triangle colors
    face_colors = np.concatenate((
        img[:-1, :-1].reshape(-1, 3),
        img[:-1, :-1].reshape(-1, 3),
        img[+1:, +1:].reshape(-1, 3),
        img[+1:, +1:].reshape(-1, 3)
    ), axis=0)

    # remove invalid faces
    if valid is not None:
        assert valid.shape == (H, W)
        valid_idxs = valid.ravel()
        valid_faces = valid_idxs[faces].all(axis=-1)
        faces = faces[valid_faces]
        face_colors = face_colors[valid_faces]

    assert len(faces) == len(face_colors)
    return dict(vertices=vertices, face_colors=face_colors, faces=faces)


def cat_meshes(meshes):
    vertices, faces, colors = zip(*[(m['vertices'], m['faces'], m['face_colors']) for m in meshes])
    n_vertices = np.cumsum([0]+[len(v) for v in vertices])
    for i in range(len(faces)):
        faces[i][:] += n_vertices[i]

    vertices = np.concatenate(vertices)
    colors = np.concatenate(colors)
    faces = np.concatenate(faces)
    return dict(vertices=vertices, face_colors=colors, faces=faces)


def show_duster_pairs(view1, view2, pred1, pred2):
    import matplotlib.pyplot as pl
    pl.ion()

    for e in range(len(view1['instance'])):
        i = view1['idx'][e]
        j = view2['idx'][e]
        img1 = rgb(view1['img'][e])
        img2 = rgb(view2['img'][e])
        conf1 = pred1['conf'][e].squeeze()
        conf2 = pred2['conf'][e].squeeze()
        score = conf1.mean()*conf2.mean()
        print(f">> Showing pair #{e} {i}-{j} {score=:g}")
        pl.clf()
        pl.subplot(221).imshow(img1)
        pl.subplot(223).imshow(img2)
        pl.subplot(222).imshow(conf1, vmin=1, vmax=30)
        pl.subplot(224).imshow(conf2, vmin=1, vmax=30)
        pts1 = pred1['pts3d'][e]
        pts2 = pred2['pts3d_in_other_view'][e]
        pl.subplots_adjust(0, 0, 1, 1, 0, 0)
        if input('show pointcloud? (y/n) ') == 'y':
            show_raw_pointcloud(cat(pts1, pts2), cat(img1, img2), point_size=5)


def auto_cam_size(im_poses):
    return 0.1 * get_med_dist_between_poses(im_poses)


class SceneViz:
    def __init__(self):
        self.scene = trimesh.Scene()

    def add_pointcloud(self, pts3d, color, mask=None):
        pts3d = to_numpy(pts3d)
        mask = to_numpy(mask)
        if mask is None:
            mask = [slice(None)] * len(pts3d)
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3))

        if isinstance(color, (list, np.ndarray, torch.Tensor)):
            color = to_numpy(color)
            col = np.concatenate([p[m] for p, m in zip(color, mask)])
            assert col.shape == pts.shape
            pct.visual.vertex_colors = uint8(col.reshape(-1, 3))
        else:
            assert len(color) == 3
            pct.visual.vertex_colors = np.broadcast_to(uint8(color), pts.shape)

        self.scene.add_geometry(pct)
        return self

    def add_camera(self, pose_c2w, focal=None, color=(0, 0, 0), image=None, imsize=None, cam_size=0.03):
        pose_c2w, focal, color, image = to_numpy((pose_c2w, focal, color, image))
        add_scene_cam(self.scene, pose_c2w, color, image, focal, screen_width=cam_size)
        return self

    def add_cameras(self, poses, focals=None, images=None, imsizes=None, colors=None, **kw):
        def get(arr, idx): return None if arr is None else arr[idx]
        for i, pose_c2w in enumerate(poses):
            self.add_camera(pose_c2w, get(focals, i), image=get(images, i),
                            color=get(colors, i), imsize=get(imsizes, i), **kw)
        return self

    def show(self, point_size=2):
        self.scene.show(line_settings={'point_size': point_size})


def show_raw_pointcloud_with_cams(imgs, pts3d, mask, focals, cams2world,
                                  point_size=2, cam_size=0.05, cam_color=None):
    """ Visualization of a pointcloud with cameras
        imgs = (N, H, W, 3) or N-size list of [(H,W,3), ...]
        pts3d = (N, H, W, 3) or N-size list of [(H,W,3), ...]
        focals = (N,) or N-size list of [focal, ...]
        cams2world = (N,4,4) or N-size list of [(4,4), ...]
    """
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
    col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
    pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
    scene.add_geometry(pct)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      imgs[i] if i < len(imgs) else None, focals[i], screen_width=cam_size)

    scene.show(line_settings={'point_size': point_size})


def add_scene_cam(scene, pose_c2w, edge_color, image=None, focal=None, imsize=None, screen_width=0.03):

    if image is not None:
        H, W, THREE = image.shape
        assert THREE == 3
        if image.dtype != np.uint8:
            image = np.uint8(255*image)
    elif imsize is not None:
        W, H = imsize
    elif focal is not None:
        H = W = focal / 1.1
    else:
        H = W = 1

    if focal is None:
        focal = min(H, W) * 1.1  # default value
    elif isinstance(focal, np.ndarray):
        focal = focal[0]

    # create fake camera
    height = focal * screen_width / H
    width = screen_width * 0.5**0.5
    rot45 = np.eye(4)
    rot45[:3, :3] = Rotation.from_euler('z', np.deg2rad(45)).as_matrix()
    rot45[2, 3] = -height  # set the tip of the cone = optical center
    aspect_ratio = np.eye(4)
    aspect_ratio[0, 0] = W/H
    transform = pose_c2w @ OPENGL @ aspect_ratio @ rot45
    cam = trimesh.creation.cone(width, height, sections=4)  # , transform=transform)

    # this is the image
    if image is not None:
        vertices = geotrf(transform, cam.vertices[[4, 5, 1, 3]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]])
        img = trimesh.Trimesh(vertices=vertices, faces=faces)
        uv_coords = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        img.visual = trimesh.visual.TextureVisuals(uv_coords, image=PIL.Image.fromarray(image))
        scene.add_geometry(img)

    # this is the camera mesh
    rot2 = np.eye(4)
    rot2[:3, :3] = Rotation.from_euler('z', np.deg2rad(2)).as_matrix()
    vertices = np.r_[cam.vertices, 0.95*cam.vertices, geotrf(rot2, cam.vertices)]
    vertices = geotrf(transform, vertices)
    faces = []
    for face in cam.faces:
        if 0 in face:
            continue
        a, b, c = face
        a2, b2, c2 = face + len(cam.vertices)
        a3, b3, c3 = face + 2*len(cam.vertices)

        # add 3 pseudo-edges
        faces.append((a, b, b2))
        faces.append((a, a2, c))
        faces.append((c2, b, c))

        faces.append((a, b, b3))
        faces.append((a, a3, c))
        faces.append((c3, b, c))

    # no culling
    faces += [(c, b, a) for a, b, c in faces]

    cam = trimesh.Trimesh(vertices=vertices, faces=faces)
    cam.visual.face_colors[:, :3] = edge_color
    scene.add_geometry(cam)


def cat(a, b):
    return np.concatenate((a.reshape(-1, 3), b.reshape(-1, 3)))


OPENGL = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])


CAM_COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (255, 204, 0), (0, 204, 204),
              (128, 255, 255), (255, 128, 255), (255, 255, 128), (0, 0, 0), (128, 128, 128)]


def uint8(colors):
    if not isinstance(colors, np.ndarray):
        colors = np.array(colors)
    if np.issubdtype(colors.dtype, np.floating):
        colors *= 255
    assert 0 <= colors.min() and colors.max() < 256
    return np.uint8(colors)


def segment_sky(image):
    import cv2
    from scipy import ndimage

    # Convert to HSV
    image = to_numpy(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.uint8(255*image.clip(min=0, max=1))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for blue color and create mask
    lower_blue = np.array([0, 0, 100])
    upper_blue = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue).view(bool)

    # add luminous gray
    mask |= (hsv[:, :, 1] < 10) & (hsv[:, :, 2] > 150)
    mask |= (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 180)
    mask |= (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 220)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask2 = ndimage.binary_opening(mask, structure=kernel)

    # keep only largest CC
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask2.view(np.uint8), connectivity=8)
    cc_sizes = stats[1:, cv2.CC_STAT_AREA]
    order = cc_sizes.argsort()[::-1]  # bigger first
    i = 0
    selection = []
    while i < len(order) and cc_sizes[order[i]] > cc_sizes[order[0]] / 2:
        selection.append(1 + order[i])
        i += 1
    mask3 = np.in1d(labels, selection).reshape(labels.shape)

    # Apply mask
    return torch.from_numpy(mask3)

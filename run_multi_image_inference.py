import os

import numpy as np
import open3d as o3d
import copy
import open3d.t.pipelines.registration as treg
import torch

from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from apple_model.apple_optimizer import AppleOptimizer
from apple_model.correspondence_loss import CorrespondenceLoss


def calculate_surface_curvature(pcd, radius=0.1, max_nn=30):
    pcd_n = copy.deepcopy(pcd)
    pcd_n.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    covs = np.asarray(pcd_n.covariances)
    vals, vecs = np.linalg.eig(covs)
    curvature = np.min(vals, axis=1) / np.sum(vals, axis=1)
    return curvature


def rotate(apple_model, target_pcd):
    axis = apple_model.axis.points
    normal = np.asarray(axis[0] - axis[1])
    unit_normal = normal / np.linalg.norm(normal)
    unit_normal = unit_normal / np.linalg.norm(unit_normal)

    old_x_axis = np.array([1, 0, 0])

    z_axis = unit_normal
    y_axis = np.cross(old_x_axis, z_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(z_axis, y_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    axis = np.stack([x_axis, y_axis, z_axis])

    p = apple_model.axis.points
    dist_pre = np.linalg.norm(np.asarray(p[0] - p[1]))
    print('dist_pre:{}'.format(dist_pre))

    apple_model.pcd.points = o3d.utility.Vector3dVector(np.dot(apple_model.pcd.points, axis.T))
    apple_model.axis.points = o3d.utility.Vector3dVector(np.dot(apple_model.axis.points, axis.T))

    o3d.visualization.draw_geometries([apple_model.pcd, apple_model.axis])

    p = apple_model.axis.points
    dist_after = np.linalg.norm(np.asarray(p[0] - p[1]))
    print('dist_after:{}'.format(dist_after))

    target_pcd.points = o3d.utility.Vector3dVector(np.dot(target_pcd.points, axis.T))

    o3d.io.write_point_cloud('fill_void/aligned.ply', apple_model.pcd + target_pcd)


if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 1000

    # model_name = "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
    model_name = "output/checkpoint-best.pth"

    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    data_root = r'D:\datasets\apple\defect_ratio_test'
    images = load_images(os.path.join(data_root, 'subset'), size=224)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
    #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
    # in each view you have:
    # an integer image identifier: view1['idx'] and view2['idx']
    # the img: view1['img'] and view2['img']
    # the image shape: view1['true_shape'] and view2['true_shape']
    # an instance string output by the dataloader: view1['instance'] and view2['instance']
    # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
    # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
    # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

    # next we'll use the global_aligner to align the predictions
    # depending on your task, you may be fine with the raw output and not need it
    # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
    # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # visualize reconstruction
    # scene.show()

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    # load mask
    image_names = sorted(os.listdir(os.path.join(data_root, 'subset')))
    masks = []
    for name in image_names:
        mask = np.load(os.path.join(data_root, 'masks_cropped', name.replace('jpg', 'npy')))
        masks.append(mask)
    pcd = o3d.geometry.PointCloud()
    points = []
    points_with_grad = []
    colors = []
    for pts, mask, image in zip(pts3d, masks, imgs):
        for row in range(pts.shape[0]):
            for column in range(pts.shape[1]):
                if mask[row][column] == 255:
                    points_with_grad.append(pts[row][column])
                    if max(image[row][column]) < 0.3:
                        colors.append([0, 0, 0])
                    else:
                        colors.append(image[row][column])

    points = np.asarray(list(map(lambda x: x.detach().cpu(), points_with_grad)))

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(k=20)
    # o3d.visualization.draw_geometries_with_editing([pcd])

    # 点云清理
    voxel_down_pcd = pcd.voxel_down_sample(0.025)
    pcd, ind = voxel_down_pcd.remove_radius_outlier(nb_points=5, radius=0.5)
    o3d.visualization.draw_geometries_with_editing([pcd])

    o3d.io.write_point_cloud('./bad_apple.ply', pcd)


    # pcd_colors = np.asarray(pcd.colors)
    # bad_idxes = []
    # for idx, color in pcd_colors:
    #     if max(colors) < 0.5:
    #         bad_idxes.ap

    radii = [0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.1]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, 0.2)
    o3d.visualization.draw_geometries([pcd, rec_mesh])

    sampled_pcd = rec_mesh.sample_points_uniformly(number_of_points=10000)
    o3d.visualization.draw_geometries_with_editing([sampled_pcd])

    mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(rec_mesh)
    mesh_tensor.fill_holes()
    o3d.visualization.draw([{'name': 'filled', 'geometry': mesh_tensor}])

    rec_mesh.get_surface_area()

    # o3d.visualization.draw_geometries([pcd])

    # curvature = calculate_surface_curvature(pcd)
    # normed_curvatue = (curvature - np.min(curvature)) / (np.max(curvature) - np.min(curvature))
    # pcd.colors = o3d.utility.Vector3dVector(np.asarray([[item//1, 0, 0] for item in normed_curvatue]))

    # o3d.visualization.draw_geometries([pcd])

    optimizer = AppleOptimizer(target=pcd)
    optimizer.opt_with_cpd()
    rotate(optimizer.optimized_model, optimizer.target_original)
    print('opt res = {}'.format(optimizer.opt_res))
    # o3d.io.write_point_cloud(r'D:\Projects\dust3r\fill_void\res.pcd', optimizer.optimized_model.pcd)
    # o3d.io.write_point_cloud(r'D:\Projects\dust3r\fill_void\target.pcd', optimizer.target_original)
    optimizer.vis_res()

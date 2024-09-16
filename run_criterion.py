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
    # viewer = o3d.visualization.Visualizer()
    # viewer.create_window()
    # for geometry in [apple_model.pcd, apple_model.axis, target_pcd]:
    #     viewer.add_geometry(geometry)
    # opt = viewer.get_render_option()
    # opt.show_coordinate_frame = True
    # viewer.run()
    # viewer.destroy_window()

    axis = apple_model.axis.points
    normal = np.asarray(axis[0] - axis[1])
    unit_normal = normal / np.linalg.norm(normal)

    old_x_axis = np.array([1, 0, 0])

    z_axis = unit_normal
    y_axis = np.cross(old_x_axis, z_axis)
    x_axis = np.cross(z_axis, y_axis)

    axis = np.stack([x_axis, y_axis, z_axis])

    apple_model.pcd.points = o3d.utility.Vector3dVector(np.dot(apple_model.pcd.points, axis.T))
    apple_model.axis.points = o3d.utility.Vector3dVector(np.dot(apple_model.axis.points, axis.T))
    target_pcd.points = o3d.utility.Vector3dVector(np.dot(target_pcd.points, axis.T))

    o3d.io.write_point_cloud('fill_void/aligned.ply', apple_model.pcd + target_pcd)

    # viewer = o3d.visualization.Visualizer()
    # viewer.create_window()
    # for geometry in [apple_model.pcd, apple_model.axis, target_pcd]:
    #     viewer.add_geometry(geometry)
    # opt = viewer.get_render_option()
    # opt.show_coordinate_frame = True
    # viewer.run()
    # viewer.destroy_window()


if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
    # model_name = "./output/0909/checkpoint-last.pth"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    images = load_images(['data/apple/Training/images/color_7.jpg', 'data/apple/Training/images/color_8.jpg'],
                         size=224)
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

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    # load mask
    mask_root = 'data/apple/Training/masks'
    mask1 = np.load(os.path.join(mask_root, 'mask_7.npy'))
    mask2 = np.load(os.path.join(mask_root, 'mask_8.npy'))
    masks = [mask1, mask2]

    pcd = o3d.geometry.PointCloud()
    points = []
    points_with_grad = []
    for pts, mask in zip(pts3d, masks):
        for i in range(pts.shape[0]):
            for j in range(pts.shape[1]):
                if mask[i][j] == 255:
                    points_with_grad.append(pts[i][j])

    points = np.asarray(list(map(lambda x: x.detach().cpu(), points_with_grad)))

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(k=20)

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

    # visualize reconstruction
    scene.show()

    # find 2D-2D matches between the two images
    from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

    pts2d_list, pts3d_list = [], []
    for i in range(2):
        conf_i = confidence_masks[i].cpu().numpy()
        pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
        pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
    reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
    print(f'found {num_matches} matches')
    matches_im1 = pts2d_list[1][reciprocal_in_P2]
    matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

    # visualize a few matches
    import numpy as np
    from matplotlib import pyplot as pl

    n_viz = 10
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
    img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    pl.show(block=True)

import pickle

import open3d as o3d
import pyvista as pv
import pymeshfix
import pymeshlab
from apple_model.ideal_model import AppleModel


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def run_surface_recon(model_path='aligned.ply'):
    # load point cloud
    pcd = o3d.io.read_point_cloud(model_path)
    # down sample and filter outliers
    pcd_dsp = pcd.voxel_down_sample(0.025)
    cl, ind = pcd_dsp.remove_radius_outlier(nb_points=10, radius=0.05)
    display_inlier_outlier(pcd_dsp, ind)
    inlier_cloud = pcd_dsp.select_by_index(ind)
    o3d.io.write_point_cloud('tmp.ply', inlier_cloud)

    mesh, _ = inlier_cloud.compute_convex_hull()
    mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])
    volume_convex_hull = o3d.geometry.TriangleMesh.get_volume(mesh)

    # surface reconstruction
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh("tmp.ply")
    ms.compute_normal_for_point_clouds()
    ms.generate_surface_reconstruction_screened_poisson()
    ms.apply_coord_laplacian_smoothing_surface_preserving(iterations=10)
    ms.meshing_remove_unreferenced_vertices()
    ms.save_current_mesh("tmp_surface.ply")

    # check water tight
    mesh = o3d.io.read_triangle_mesh('tmp_surface.ply')
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
    print('is water_tight: {}'.format(mesh.is_watertight()))
    try:
        volume = o3d.geometry.TriangleMesh.get_volume(mesh)
        print('Volume from Pos = {}'.format(volume))
        return volume
    except Exception as e:
        print('Volume = {}'.format(volume_convex_hull))
        return volume_convex_hull


if __name__ == '__main__':
    run_surface_recon()

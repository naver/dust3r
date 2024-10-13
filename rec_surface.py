import open3d as o3d
import numpy as np
from run_multi_image_inference import rotate
from apple_model.apple_optimizer import AppleOptimizer
import pymeshlab

if __name__ == "__main__":
    mesh_path = r"C:\Users\douba\Desktop\cropped_1_warpped_color.ply"
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd = mesh.sample_points_uniformly(number_of_points=10000)
    o3d.visualization.draw_geometries([pcd])
    bad_count = 0
    new_colors = []
    for idx, color in enumerate(np.asarray(pcd.colors)):
        if max(color) < 0.3:
            bad_count += 1
            new_colors.append((0, 0, 1))
        else:
            new_colors.append(color)
    pcd.colors = o3d.utility.Vector3dVector(new_colors)
    bad_ratio = bad_count / len(new_colors)
    print('bad ratio = {}'.format(bad_ratio))

    surface_area = mesh.get_surface_area()
    print('surface area = {}'.format(surface_area))
    print('bad area = {}'.format(surface_area * bad_ratio))
    o3d.visualization.draw_geometries([pcd])

    optimizer = AppleOptimizer(target=pcd)
    optimizer.opt_with_cpd()
    rotate(optimizer.optimized_model, optimizer.target_original)
    print('opt res = {}'.format(optimizer.opt_res))
    optimizer.vis_res()

    color = [(1, 0, 0) for _ in np.asarray(optimizer.optimized_model.pcd.points)]
    optimizer.optimized_model.colors = o3d.utility.Vector3dVector(color)

    tmp_dir = r'C:\Users\douba\Desktop\aligned_target_and_src.ply'
    o3d.io.write_point_cloud(tmp_dir, optimizer.optimized_model.pcd)

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(tmp_dir)
    ms.compute_normal_for_point_clouds()
    ms.generate_surface_reconstruction_screened_poisson()
    ms.apply_coord_laplacian_smoothing_surface_preserving(iterations=10)
    ms.meshing_remove_unreferenced_vertices()
    ms.save_current_mesh("tmp_surface.ply")

    rec_mesh = o3d.io.read_triangle_mesh('tmp_surface.ply')
    rec_mesh.compute_vertex_normals()
    total_area = rec_mesh.get_surface_area()
    print('total area = {}'.format(total_area))
    print('bad area vs total area = {}'.format(bad_ratio * surface_area / total_area))
    o3d.visualization.draw_geometries([rec_mesh])
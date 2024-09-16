import open3d as o3d


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


if __name__ == '__main__':
    res_pcd = o3d.io.read_point_cloud(r'D:\Projects\dust3r\fill_void\res.pcd')
    target_pcd = o3d.io.read_point_cloud(r'D:\Projects\dust3r\fill_void\target.pcd')
    o3d.visualization.draw_geometries([res_pcd, target_pcd])

    total_pcd = res_pcd + target_pcd
    total_pcd_down_sampled = total_pcd.voxel_down_sample(0.05)

    o3d.visualization.draw_geometries([total_pcd_down_sampled])

    cl, ind = total_pcd_down_sampled.remove_radius_outlier(nb_points=4, radius=0.05)
    display_inlier_outlier(total_pcd_down_sampled, ind)

    pcd_radius_down_sampled = total_pcd_down_sampled.select_by_index(ind)
    o3d.visualization.draw_geometries([pcd_radius_down_sampled])

    total_pcd.estimate_normals()
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            total_pcd, depth=5)
    o3d.visualization.draw_geometries([mesh])
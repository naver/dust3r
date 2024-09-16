import open3d as o3d
import pyvista
import pyvista as pv
import pymeshfix


if __name__ == '__main__':
    mesh_in = pyvista.read('./matching_result_without_lines.ply')
    print('reconstructing')
    surf = mesh_in.reconstruct_surface(nbr_sz=20, sample_spacing=2)

    print('repairing')
    mf = pymeshfix.MeshFix(surf)
    mf.repair()
    repaired = mf.mesh

    print('plotting')
    pl = pv.Plotter()
    pl.add_mesh(mesh_in, color='k', point_size=10)
    pl.add_mesh(repaired)
    pl.add_title('Reconstructed Surface')
    pl.show()



    # mesh_in = o3d.io.read_triangle_mesh('./surface.ply')
    # mesh_in.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh_in])
    #
    # voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 32
    # print(f'voxel_size = {voxel_size:e}')
    # mesh_smp = mesh_in.simplify_vertex_clustering(
    #     voxel_size=voxel_size,
    #     contraction=o3d.geometry.SimplificationContraction.Average)
    # print(
    #     f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
    # )
    # print(mesh_smp.is_watertight())
    # o3d.visualization.draw_geometries([mesh_smp])
    # v = o3d.geometry.TriangleMesh.get_volume(mesh_smp)
    # print(v)
    #
    # mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=10)
    # mesh_out.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh_out])
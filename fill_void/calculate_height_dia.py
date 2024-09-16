import open3d as o3d
import numpy as np

if __name__ == '__main__':
    mesh = o3d.io.read_triangle_mesh('tmp_surface.ply')
    mesh.compute_vertex_normals()
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox.color = (0, 1, 0)
    # o3d.visualization.draw_geometries([mesh, bbox])
    dia1, dia2, height = bbox.max_bound - bbox.min_bound
    print('dia1 = {}, dia2 = {}, height = {}'.format(dia1, dia2, height))

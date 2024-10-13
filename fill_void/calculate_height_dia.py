import open3d as o3d


def run_cal_height_dia(model_path='tmp_surface.ply'):
    mesh = o3d.io.read_triangle_mesh(model_path)
    mesh.compute_vertex_normals()
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox.color = (0, 1, 0)
    o3d.visualization.draw_geometries([mesh, bbox])
    dia1, dia2, height = bbox.max_bound - bbox.min_bound
    print('dia1 = {}, dia2 = {}, height = {}'.format(dia1, dia2, height))
    return dia1, dia2, height


if __name__ == '__main__':
    run_cal_height_dia()

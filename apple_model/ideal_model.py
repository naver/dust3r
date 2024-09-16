import pdb

import open3d as o3d
import numpy as np
import pickle


class AppleModel:
    def __init__(self, params, sample_step, v_range=1.0):
        if isinstance(params, list):
            self.a, self.b, self.c, self.p1, self.p2 = params
        elif isinstance(params, np.ndarray):
            self.a, self.b, self.c, self.p1, self.p2 = list(params)
        self.pcd = o3d.geometry.PointCloud()
        self.uv = []
        self.valid_idx = []
        self.v_range = v_range
        # u_pos = np.concatenate([-self.powspace(0, np.pi / 2,  0.7, sample_step//2)[::-1],
        #                         self.powspace(0, np.pi / 2,  0.7, sample_step//2)])
        u_pos = np.linspace(-np.pi / 2, np.pi / 2, num=sample_step)
        for u in u_pos:
            for v in np.linspace(0, 2 * np.pi, num=sample_step):
                if v <= v_range * 2 * np.pi:
                    self.valid_idx.append(len(self.uv))
                self.uv.append((u, v))
        self.get_apple_points()
        self.axis = o3d.geometry.LineSet()
        self.axis.points = o3d.utility.Vector3dVector(np.asarray([self.pcd.points[1], self.pcd.points[-1]]))
        self.axis.lines = o3d.utility.Vector2iVector(np.asarray([[0, 1]]))

    @staticmethod
    def powspace(start, stop, power, num):
        start = np.power(start, 1 / float(power))
        stop = np.power(stop, 1 / float(power))
        return np.power(np.linspace(start, stop, num=num), power)

    def get_apple_point(self, u, v):
        x = self.a * np.cos(u) * np.cos(v)
        y = self.b * np.cos(u) * np.sin(v)
        z = self.c * np.sin(u)
        return x, y, z

    def add_disturb(self):
        for idx, ((u, v), point, normal) in enumerate(zip(self.uv, self.pcd.points, self.pcd.normals)):
            g_uv = -self.p1 * (np.e ** (-2 * u)) - self.p2 * (np.e ** (2 * u))
            self.pcd.points[idx] += g_uv * normal

    def get_apple_points(self):
        points = []
        for u, v in self.uv:
            points.append(self.get_apple_point(u, v))
        points = np.asarray(points)
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.estimate_normals()
        self.pcd.orient_normals_consistent_tangent_plane(k=50)
        self.add_disturb()
        self.remove_partial()
        # self.remove_outliers()

    def remove_partial(self):
        self.pcd = self.pcd.select_by_index(self.valid_idx)

    def remove_outliers(self):
        voxel_down_pcd = self.pcd.voxel_down_sample(voxel_size=0.01)
        self.pcd, ind = voxel_down_pcd.remove_radius_outlier(nb_points=1, radius=0.1)

    def vis_points(self):
        o3d.visualization.draw_geometries([self.pcd])


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


if __name__ == '__main__':
    param = np.asarray([0.51265244, 0.51242454, 0.53797388, 0.01, 0.01])
    apple = AppleModel(params=param, sample_step=30, v_range=1)
    o3d.visualization.draw_geometries([apple.pcd])
    voxel_down_pcd = apple.pcd.voxel_down_sample(voxel_size=0.01)
    cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=2, radius=0.1)
    display_inlier_outlier(voxel_down_pcd, ind)
    o3d.visualization.draw_geometries([cl])

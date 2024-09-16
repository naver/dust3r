import pdb

import numpy
import scipy
import copy
import numpy as np
import open3d as o3d
import open3d.t.pipelines.registration as treg
import pycpd

from apple_model.ideal_model import AppleModel


class AppleOptimizer:
    def __init__(self, target):
        self.target_original = copy.deepcopy(target)
        self.target = target.voxel_down_sample(0.05)
        self.init_param = np.asarray([0.51265244, 0.51242454, 0.51235608, 0.01, 0.01])
        self.bounds = [(0, 1), (0, 1), (0, 1), (0, 0.1), (0, 0.1)]
        self.opt_res = None
        self.optimized_model = None
        self.reg_res = None
        self.reg_dict = {}
        self.max_correspondence_distance = 0.025
        self.dense_grid_size = 100
        self.sparse_grid_size = 30
        self.partial_ratio = 0.65

    @staticmethod
    def draw_registration_result(source, target, transformation=None, reg_res=None):

        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        data_to_vis = [source_temp, target_temp]
        if transformation is not None:
            source_temp.transform(transformation)
        if reg_res is not None:
            line_set = o3d.geometry.LineSet().create_from_point_cloud_correspondences(cloud0=source_temp,
                                                                                      cloud1=target_temp,
                                                                                      correspondences=np.asarray(
                                                                                          reg_res.correspondence_set))
            line_set.paint_uniform_color([1, 0, 0])
            data_to_vis.append(line_set)
        o3d.visualization.draw_geometries(data_to_vis)

    def vis_res(self):
        if self.reg_res is None:
            print('Registration Results Not found')
        self.draw_registration_result(self.optimized_model.pcd, self.target_original, reg_res=self.reg_res)

    def cpd_align(self, source):
        # create a RigidRegistration object
        reg = pycpd.RigidRegistration(X=np.asarray(self.target.points), Y=np.asarray(source.points), w=0.2,
                                      max_iterations=500)
        # run the registration & collect the results
        TY, trans = reg.register()
        return TY, trans, reg

    def cpd_objective(self, inp):
        # CPD register with current parameter
        source_model = AppleModel(inp, sample_step=self.sparse_grid_size, v_range=self.partial_ratio)
        inp = tuple(inp)
        try:
            if inp in self.reg_dict:
                reg = self.reg_dict[inp]
                transformed_source = reg.transform_point_cloud(Y=np.asarray(source_model.pcd))
            else:
                transformed_source, transformation, reg = self.cpd_align(source_model.pcd)
                self.reg_dict[inp] = reg
            source_model.pcd.points = o3d.utility.Vector3dVector(transformed_source)
            reg_res = o3d.pipelines.registration.evaluate_registration(source_model.pcd, self.target,
                                                                       self.max_correspondence_distance,
                                                                       transformation=numpy.eye(4))
        except Exception as e:
            print('Objective Calculation Error: {}'.format(e))
            return 5.0
        print('input parameters = {}'.format(reg_res))
        return reg_res.inlier_rmse

    def opt_with_cpd(self):
        print('Optimization with CPD Begin')

        options = {'maxiter': 1}
        res = scipy.optimize.minimize(self.cpd_objective, x0=self.init_param, method='Nelder-Mead', tol=1e-2,
                                      bounds=self.bounds, options=options)
        print('Optimization Finished, x = {}'.format(res.x))

        self.opt_res = res.x
        self.optimized_model = AppleModel(self.opt_res, sample_step=self.dense_grid_size)

        # self.optimized_model = AppleModel(self.opt_res, sample_step=self.sparse_grid_size, v_range=self.partial_ratio)

        reg = self.reg_dict[tuple(self.opt_res)]
        transformed_source = reg.transform_point_cloud(Y=np.asarray(self.optimized_model.pcd.points))
        transformed_axis = reg.transform_point_cloud(Y=np.asarray(self.optimized_model.axis.points))
        self.optimized_model.pcd.points = o3d.utility.Vector3dVector(transformed_source)
        self.optimized_model.axis.points = o3d.utility.Vector3dVector(transformed_axis)
        self.reg_res = o3d.pipelines.registration.evaluate_registration(self.optimized_model.pcd,
                                                                        self.target_original,
                                                                        self.max_correspondence_distance,
                                                                        transformation=numpy.eye(4))
        print('final reg results = {}'.format(self.reg_res))
        o3d.visualization.draw_geometries([self.optimized_model.pcd, self.target_original, self.optimized_model.axis])
        # self.draw_registration_result(self.optimized_model.pcd, self.target_original, reg_res=self.reg_res)

import collections

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import open3d as o3d

from apple_model.apple_optimizer import AppleOptimizer


class CorrespondenceLoss(nn.Module):
    def __init__(self, fitness_threshold=0.3):
        super().__init__()
        self.loos_func = nn.MSELoss()
        self.fitness_threshold = fitness_threshold

    def forward(self, points_t):
        points = np.asarray(list(map(lambda x: x.detach().cpu(), points_t)))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(k=20)

        optimizer = AppleOptimizer(target=pcd)
        optimizer.opt_with_cpd()
        reg_res = optimizer.reg_res
        loss = torch.zeros(1)
        if optimizer.reg_res.fitness < self.fitness_threshold:
            return loss
        correspondence = np.asarray(reg_res.correspondence_set)
        matching_dict = collections.defaultdict(list)
        inp = []
        target = []
        # source from ideal model, target from prediction
        for source_idx, target_idx in correspondence:
            matching_dict[target_idx].append(optimizer.optimized_model.pcd.points[source_idx])
        for target_idx, p_list in matching_dict.items():
            inp.append(points_t[target_idx].cpu())
            target.append(np.mean(np.asarray(p_list), axis=0))
        loss += self.loos_func(torch.tensor(inp), torch.tensor(target))
        return loss



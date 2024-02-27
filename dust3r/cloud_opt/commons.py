# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utility functions for global alignment
# --------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np


def edge_str(i, j):
    return f'{i}_{j}'


def i_j_ij(ij):
    return edge_str(*ij), ij


def edge_conf(conf_i, conf_j, edge):
    return float(conf_i[edge].mean() * conf_j[edge].mean())


def compute_edge_scores(edges, conf_i, conf_j):
    return {(i, j): edge_conf(conf_i, conf_j, e) for e, (i, j) in edges}


def NoGradParamDict(x):
    assert isinstance(x, dict)
    return nn.ParameterDict(x).requires_grad_(False)


def get_imshapes(edges, pred_i, pred_j):
    n_imgs = max(max(e) for e in edges) + 1
    imshapes = [None] * n_imgs
    for e, (i, j) in enumerate(edges):
        shape_i = tuple(pred_i[e].shape[0:2])
        shape_j = tuple(pred_j[e].shape[0:2])
        if imshapes[i]:
            assert imshapes[i] == shape_i, f'incorrect shape for image {i}'
        if imshapes[j]:
            assert imshapes[j] == shape_j, f'incorrect shape for image {j}'
        imshapes[i] = shape_i
        imshapes[j] = shape_j
    return imshapes


def get_conf_trf(mode):
    if mode == 'log':
        def conf_trf(x): return x.log()
    elif mode == 'sqrt':
        def conf_trf(x): return x.sqrt()
    elif mode == 'm1':
        def conf_trf(x): return x-1
    elif mode in ('id', 'none'):
        def conf_trf(x): return x
    else:
        raise ValueError(f'bad mode for {mode=}')
    return conf_trf


def l2_dist(a, b, weight):
    return ((a - b).square().sum(dim=-1) * weight)


def l1_dist(a, b, weight):
    return ((a - b).norm(dim=-1) * weight)


ALL_DISTS = dict(l1=l1_dist, l2=l2_dist)


def signed_log1p(x):
    sign = torch.sign(x)
    return sign * torch.log1p(torch.abs(x))


def signed_expm1(x):
    sign = torch.sign(x)
    return sign * torch.expm1(torch.abs(x))


def cosine_schedule(t, lr_start, lr_end):
    assert 0 <= t <= 1
    return lr_end + (lr_start - lr_end) * (1+np.cos(t * np.pi))/2


def linear_schedule(t, lr_start, lr_end):
    assert 0 <= t <= 1
    return lr_start + (lr_end - lr_start) * t

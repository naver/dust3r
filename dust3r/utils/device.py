# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions for DUSt3R
# --------------------------------------------------------
import numpy as np
import torch


def todevice(batch, device, callback=None, non_blocking=False):
    ''' Transfer some variables to another device (i.e. GPU, CPU:torch, CPU:numpy).

    batch: list, tuple, dict of tensors or other things
    device: pytorch device or 'numpy'
    callback: function that would be called on every sub-elements.
    '''
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {k: todevice(v, device) for k, v in batch.items()}

    if isinstance(batch, (tuple, list)):
        return type(batch)(todevice(x, device) for x in batch)

    x = batch
    if device == 'numpy':
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x


to_device = todevice  # alias


def to_numpy(x): return todevice(x, 'numpy')
def to_cpu(x): return todevice(x, 'cpu')
def to_cuda(x): return todevice(x, 'cuda')


def collate_with_cat(whatever, lists=False):
    if isinstance(whatever, dict):
        return {k: collate_with_cat(vals, lists=lists) for k, vals in whatever.items()}

    elif isinstance(whatever, (tuple, list)):
        if len(whatever) == 0:
            return whatever
        elem = whatever[0]
        T = type(whatever)

        if elem is None:
            return None
        if isinstance(elem, (bool, float, int, str)):
            return whatever
        if isinstance(elem, tuple):
            return T(collate_with_cat(x, lists=lists) for x in zip(*whatever))
        if isinstance(elem, dict):
            return {k: collate_with_cat([e[k] for e in whatever], lists=lists) for k in elem}

        if isinstance(elem, torch.Tensor):
            return listify(whatever) if lists else torch.cat(whatever)
        if isinstance(elem, np.ndarray):
            return listify(whatever) if lists else torch.cat([torch.from_numpy(x) for x in whatever])

        # otherwise, we just chain lists
        return sum(whatever, T())


def listify(elems):
    return [x for e in elems for x in e]

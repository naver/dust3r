# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Random sampling under a constraint
# --------------------------------------------------------
import numpy as np
import torch


class BatchedRandomSampler:
    """ Random sampling under a constraint: each sample in the batch has the same feature, 
    which is chosen randomly from a known pool of 'features' for each batch.

    For instance, the 'feature' could be the image aspect-ratio.

    The index returned is a tuple (sample_idx, feat_idx).
    This sampler ensures that each series of `batch_size` indices has the same `feat_idx`.
    """

    def __init__(self, dataset, batch_size, pool_size, world_size=1, rank=0, drop_last=True):
        self.batch_size = batch_size
        self.pool_size = pool_size

        self.len_dataset = N = len(dataset)
        self.total_size = round_by(N, batch_size*world_size) if drop_last else N
        assert world_size == 1 or drop_last, 'must drop the last batch in distributed mode'

        # distributed sampler
        self.world_size = world_size
        self.rank = rank
        self.epoch = None

    def __len__(self):
        return self.total_size // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # prepare RNG
        if self.epoch is None:
            assert self.world_size == 1 and self.rank == 0, 'use set_epoch() if distributed mode is used'
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)

        # random indices (will restart from 0 if not drop_last)
        sample_idxs = np.arange(self.total_size)
        rng.shuffle(sample_idxs)

        # random feat_idxs (same across each batch)
        n_batches = (self.total_size+self.batch_size-1) // self.batch_size
        feat_idxs = rng.integers(self.pool_size, size=n_batches)
        feat_idxs = np.broadcast_to(feat_idxs[:, None], (n_batches, self.batch_size))
        feat_idxs = feat_idxs.ravel()[:self.total_size]

        # put them together
        idxs = np.c_[sample_idxs, feat_idxs]  # shape = (total_size, 2)

        # Distributed sampler: we select a subset of batches
        # make sure the slice for each node is aligned with batch_size
        size_per_proc = self.batch_size * ((self.total_size + self.world_size *
                                           self.batch_size-1) // (self.world_size * self.batch_size))
        idxs = idxs[self.rank*size_per_proc: (self.rank+1)*size_per_proc]

        yield from (tuple(idx) for idx in idxs)


def round_by(total, multiple, up=False):
    if up:
        total = total + multiple-1
    return (total//multiple) * multiple

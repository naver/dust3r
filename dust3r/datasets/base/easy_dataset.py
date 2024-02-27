# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# A dataset base class that you can easily resize and combine.
# --------------------------------------------------------
import numpy as np
from dust3r.datasets.base.batched_sampler import BatchedRandomSampler


class EasyDataset:
    """ a dataset that you can easily resize and combine.
    Examples:
    ---------
        2 * dataset ==> duplicate each element 2x

        10 @ dataset ==> set the size to 10 (random sampling, duplicates if necessary)

        dataset1 + dataset2 ==> concatenate datasets
    """

    def __add__(self, other):
        return CatDataset([self, other])

    def __rmul__(self, factor):
        return MulDataset(factor, self)

    def __rmatmul__(self, factor):
        return ResizedDataset(factor, self)

    def set_epoch(self, epoch):
        pass  # nothing to do by default

    def make_sampler(self, batch_size, shuffle=True, world_size=1, rank=0, drop_last=True):
        if not (shuffle):
            raise NotImplementedError()  # cannot deal yet
        num_of_aspect_ratios = len(self._resolutions)
        return BatchedRandomSampler(self, batch_size, num_of_aspect_ratios, world_size=world_size, rank=rank, drop_last=drop_last)


class MulDataset (EasyDataset):
    """ Artifically augmenting the size of a dataset.
    """
    multiplicator: int

    def __init__(self, multiplicator, dataset):
        assert isinstance(multiplicator, int) and multiplicator > 0
        self.multiplicator = multiplicator
        self.dataset = dataset

    def __len__(self):
        return self.multiplicator * len(self.dataset)

    def __repr__(self):
        return f'{self.multiplicator}*{repr(self.dataset)}'

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, other = idx
            return self.dataset[idx // self.multiplicator, other]
        else:
            return self.dataset[idx // self.multiplicator]

    @property
    def _resolutions(self):
        return self.dataset._resolutions


class ResizedDataset (EasyDataset):
    """ Artifically changing the size of a dataset.
    """
    new_size: int

    def __init__(self, new_size, dataset):
        assert isinstance(new_size, int) and new_size > 0
        self.new_size = new_size
        self.dataset = dataset

    def __len__(self):
        return self.new_size

    def __repr__(self):
        size_str = str(self.new_size)
        for i in range((len(size_str)-1) // 3):
            sep = -4*i-3
            size_str = size_str[:sep] + '_' + size_str[sep:]
        return f'{size_str} @ {repr(self.dataset)}'

    def set_epoch(self, epoch):
        # this random shuffle only depends on the epoch
        rng = np.random.default_rng(seed=epoch+777)

        # shuffle all indices
        perm = rng.permutation(len(self.dataset))

        # rotary extension until target size is met
        shuffled_idxs = np.concatenate([perm] * (1 + (len(self)-1) // len(self.dataset)))
        self._idxs_mapping = shuffled_idxs[:self.new_size]

        assert len(self._idxs_mapping) == self.new_size

    def __getitem__(self, idx):
        assert hasattr(self, '_idxs_mapping'), 'You need to call dataset.set_epoch() to use ResizedDataset.__getitem__()'
        if isinstance(idx, tuple):
            idx, other = idx
            return self.dataset[self._idxs_mapping[idx], other]
        else:
            return self.dataset[self._idxs_mapping[idx]]

    @property
    def _resolutions(self):
        return self.dataset._resolutions


class CatDataset (EasyDataset):
    """ Concatenation of several datasets 
    """

    def __init__(self, datasets):
        for dataset in datasets:
            assert isinstance(dataset, EasyDataset)
        self.datasets = datasets
        self._cum_sizes = np.cumsum([len(dataset) for dataset in datasets])

    def __len__(self):
        return self._cum_sizes[-1]

    def __repr__(self):
        # remove uselessly long transform
        return ' + '.join(repr(dataset).replace(',transform=Compose( ToTensor() Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))', '') for dataset in self.datasets)

    def set_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_epoch(epoch)

    def __getitem__(self, idx):
        other = None
        if isinstance(idx, tuple):
            idx, other = idx

        if not (0 <= idx < len(self)):
            raise IndexError()

        db_idx = np.searchsorted(self._cum_sizes, idx, 'right')
        dataset = self.datasets[db_idx]
        new_idx = idx - (self._cum_sizes[db_idx - 1] if db_idx > 0 else 0)

        if other is not None:
            new_idx = (new_idx, other)
        return dataset[new_idx]

    @property
    def _resolutions(self):
        resolutions = self.datasets[0]._resolutions
        for dataset in self.datasets[1:]:
            assert tuple(dataset._resolutions) == tuple(resolutions)
        return resolutions

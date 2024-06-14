# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# AachenDayNight dataloader
# --------------------------------------------------------
import os
from dust3r_visloc.datasets.base_colmap import BaseVislocColmapDataset


class VislocAachenDayNight(BaseVislocColmapDataset):
    def __init__(self, root, subscene, pairsfile, topk=1, cache_sfm=False):
        assert subscene in [None, '', 'day', 'night', 'all']
        self.subscene = subscene
        image_path = os.path.join(root, 'images')
        map_path = os.path.join(root, 'mapping/colmap/reconstruction')
        query_path = os.path.join(root, 'kapture', 'query')
        pairsfile_path = os.path.join(root, 'pairsfile/query', pairsfile + '.txt')
        super().__init__(image_path=image_path, map_path=map_path,
                         query_path=query_path, pairsfile_path=pairsfile_path,
                         topk=topk, cache_sfm=cache_sfm)
        self.scenes = [filename for filename in self.scenes if filename in self.pairs]
        if self.subscene == 'day' or self.subscene == 'night':
            self.scenes = [filename for filename in self.scenes if self.subscene in filename]

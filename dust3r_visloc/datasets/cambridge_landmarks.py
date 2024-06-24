# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Cambridge Landmarks dataloader
# --------------------------------------------------------
import os
from dust3r_visloc.datasets.base_colmap import BaseVislocColmapDataset


class VislocCambridgeLandmarks (BaseVislocColmapDataset):
    def __init__(self, root, subscene, pairsfile, topk=1, cache_sfm=False):
        image_path = os.path.join(root, subscene)
        map_path = os.path.join(root, 'mapping', subscene, 'colmap/reconstruction')
        query_path = os.path.join(root, 'kapture', subscene, 'query')
        pairsfile_path = os.path.join(root, subscene, 'pairsfile/query', pairsfile + '.txt')
        super().__init__(image_path=image_path, map_path=map_path,
                         query_path=query_path, pairsfile_path=pairsfile_path,
                          topk=topk, cache_sfm=cache_sfm)
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Base class
# --------------------------------------------------------
class BaseVislocDataset:
    def __init__(self):
        pass

    def set_resolution(self, model):
        self.maxdim = max(model.patch_embed.img_size)
        self.patch_size = model.patch_embed.patch_size

    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, idx):
        raise NotImplementedError()
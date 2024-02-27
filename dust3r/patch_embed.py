# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# PatchEmbed implementation for DUST3R,
# in particular ManyAR_PatchEmbed that Handle images with non-square aspect ratio
# --------------------------------------------------------
import torch
import dust3r.utils.path_to_croco  # noqa: F401
from models.blocks import PatchEmbed  # noqa


def get_patch_embed(patch_embed_cls, img_size, patch_size, enc_embed_dim):
    assert patch_embed_cls in ['PatchEmbedDust3R', 'ManyAR_PatchEmbed']
    patch_embed = eval(patch_embed_cls)(img_size, patch_size, 3, enc_embed_dim)
    return patch_embed


class PatchEmbedDust3R(PatchEmbed):
    def forward(self, x, **kw):
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos


class ManyAR_PatchEmbed (PatchEmbed):
    """ Handle images with non-square aspect ratio.
        All images in the same batch have the same aspect ratio.
        true_shape = [(height, width) ...] indicates the actual shape of each image.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        self.embed_dim = embed_dim
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)

    def forward(self, img, true_shape):
        B, C, H, W = img.shape
        assert W >= H, f'img should be in landscape mode, but got {W=} {H=}'
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        assert true_shape.shape == (B, 2), f"true_shape has the wrong shape={true_shape.shape}"

        # size expressed in tokens
        W //= self.patch_size[0]
        H //= self.patch_size[1]
        n_tokens = H * W

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        # allocate result
        x = img.new_zeros((B, n_tokens, self.embed_dim))
        pos = img.new_zeros((B, n_tokens, 2), dtype=torch.int64)

        # linear projection, transposed if necessary
        x[is_landscape] = self.proj(img[is_landscape]).permute(0, 2, 3, 1).flatten(1, 2).float()
        x[is_portrait] = self.proj(img[is_portrait].swapaxes(-1, -2)).permute(0, 2, 3, 1).flatten(1, 2).float()

        pos[is_landscape] = self.position_getter(1, H, W, pos.device)
        pos[is_portrait] = self.position_getter(1, W, H, pos.device)

        x = self.norm(x)
        return x, pos

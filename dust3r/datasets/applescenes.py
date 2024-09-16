import os.path
import os.path as osp
import cv2
import numpy as np
import PIL

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class AppleScenes(BaseStereoViewDataset):
    def __init__(self, *args, split, root, **kwargs):
        super().__init__(*args, **kwargs)
        if split == "train":
            self.split = "Training"
        elif split == "test":
            self.split = "Test"
        else:
            raise ValueError("")
        self.root = root
        self.dataset_label = 'Apple'
        self._load_data()

    def _load_data(self):
        # 加载苹果数据集split中的数据文件， 每行：scene_id, img_idx1, img_idx2
        split_file_path = os.path.join(self.root, 'split', '{}.csv'.format(self.split))
        with open(split_file_path, 'r') as f:
            data_idx = f.readlines()
        self.pairs = list(map(lambda x: list(map(int, x.split(','))), data_idx))

    def __len__(self):
        return len(self.pairs)

    def _get_views(self, idx, resolution, rng):
        views = []
        scene_id, img_idx1, img_idx2 = self.pairs[idx]
        for idx in [img_idx1, img_idx2]:
            image_path = osp.join(self.root, 'images', "color_{}.jpg".format(idx))
            image = imread_cv2(image_path)
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            mask_path = osp.join(self.root, 'masks', "mask_{}.npy".format(idx))
            mask = np.load(mask_path)

            views.append(dict(
                img=image,
                mask=mask,
                dataset=self.dataset_label,
                instance=image_path,
            ))
        return views



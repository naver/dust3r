#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Script to export the list of scenes for habitat (after having rendered them).
# Usage:
# python3 datasets_preprocess/preprocess_co3d.py --root data/habitat_processed
# --------------------------------------------------------
import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm


def find_all_scenes(habitat_root, n_scenes=[100000]):
    np.random.seed(777)

    try:
        fpath = os.path.join(habitat_root, f'Habitat_all_scenes.txt')
        list_subscenes = open(fpath).read().splitlines()

    except IOError:
        if input('parsing sub-folders to find scenes? (y/n) ') != 'y':
            return
        list_subscenes = []
        for root, dirs, files in tqdm(os.walk(habitat_root)):
            for f in files:
                if not f.endswith('_1_depth.exr'):
                    continue
                scene = os.path.join(os.path.relpath(root, habitat_root), f.replace('_1_depth.exr', ''))
                if hash(scene) % 1000 == 0:
                    print('... adding', scene)
                list_subscenes.append(scene)

        with open(fpath, 'w') as f:
            f.write('\n'.join(list_subscenes))
        print(f'>> wrote {fpath}')

    print(f'Loaded {len(list_subscenes)} sub-scenes')

    # separate scenes
    list_scenes = defaultdict(list)
    for scene in list_subscenes:
        scene, id = os.path.split(scene)
        list_scenes[scene].append(id)

    list_scenes = list(list_scenes.items())
    print(f'from {len(list_scenes)} scenes in total')

    np.random.shuffle(list_scenes)
    train_scenes = list_scenes[len(list_scenes) // 10:]
    val_scenes = list_scenes[:len(list_scenes) // 10]

    def write_scene_list(scenes, n, fpath):
        sub_scenes = [os.path.join(scene, id) for scene, ids in scenes for id in ids]
        np.random.shuffle(sub_scenes)

        if len(sub_scenes) < n:
            return

        with open(fpath, 'w') as f:
            f.write('\n'.join(sub_scenes[:n]))
        print(f'>> wrote {fpath}')

    for n in n_scenes:
        write_scene_list(train_scenes, n, os.path.join(habitat_root, f'Habitat_{n}_scenes_train.txt'))
        write_scene_list(val_scenes, n // 10, os.path.join(habitat_root, f'Habitat_{n//10}_scenes_val.txt'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--n_scenes", nargs='+', default=[1_000, 10_000, 100_000, 1_000_000], type=int)

    args = parser.parse_args()
    find_all_scenes(args.root, args.n_scenes)

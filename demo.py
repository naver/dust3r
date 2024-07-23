#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dust3r gradio demo executable
# --------------------------------------------------------
import os
import torch
import tempfile

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.demo import get_args_parser, main_demo, set_print_with_timestamp

import matplotlib.pyplot as pl
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    set_print_with_timestamp()

    if args.tmp_dir is not None:
        tmp_path = args.tmp_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)

    # dust3r will write the 3D model inside tmpdirname
    with tempfile.TemporaryDirectory(suffix='dust3r_gradio_demo') as tmpdirname:
        if not args.silent:
            print('Outputing stuff in', tmpdirname)
        main_demo(tmpdirname, model, args.device, args.image_size, server_name, args.server_port, silent=args.silent)

#!/bin/bash

set -e

DEVICE=${DEVICE:-cuda}

# Initialize Conda to make the `conda` command available in this script
. /opt/conda/etc/profile.d/conda.sh
conda activate dust3r

exec python3 demo.py --weights checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth --device "$DEVICE" --local_network "$@"

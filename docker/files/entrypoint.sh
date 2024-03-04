#!/bin/bash

set -eux

DEVICE=${DEVICE:-cuda}
MODEL=${MODEL:-DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}

# Initialize Conda to make the `conda` command available in this script
. /opt/conda/etc/profile.d/conda.sh
conda activate dust3r

exec python3 demo.py --weights "checkpoints/$MODEL" --device "$DEVICE" --local_network "$@"

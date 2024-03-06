#!/bin/bash

set -eux

DEVICE=${DEVICE:-cuda}
MODEL=${MODEL:-DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}

exec python3 demo.py --weights "checkpoints/$MODEL" --device "$DEVICE" --local_network "$@"

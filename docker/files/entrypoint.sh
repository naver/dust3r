#!/bin/bash

set -eux

DEVICE=${DEVICE:-cuda}
MODEL=${MODEL:-DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}

# Keep the container running for debugging
tail -f /dev/null

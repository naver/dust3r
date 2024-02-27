# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R repo root import
# --------------------------------------------------------

import sys
import os.path as path
HERE_PATH = path.normpath(path.dirname(__file__))
DUST3R_REPO_PATH = path.normpath(path.join(HERE_PATH, '../'))
# workaround for sibling import
sys.path.insert(0, DUST3R_REPO_PATH)

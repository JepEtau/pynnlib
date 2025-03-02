# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pprint import pprint
import sys
from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra

if not GlobalHydra.instance().is_initialized():
    initialize_config_module("efficient_track_anything", version_base="1.2")
    for e, v in sys.modules.items():
        if "efficient_track_anything" in e:
            print(f"{e}: {v}")

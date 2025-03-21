'''
 Copyright 2023 xtudbxk
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''




import os
curdir = os.path.split(__file__)[0]

import torch
from torch.utils.cpp_extension import load

# if torch.cuda.is_available():

#     TMPWrapper = load(name="TemporalMotionPropagation",
#                        sources=[os.path.join(curdir, "tmpwrapper.cpp"),
#                                 os.path.join(curdir, "tmp.cu")],
#                        verbose=True)

#     def tmp(feat, feat_pre, offsets, offsets_pre, distance, iters_t=30, sigma=30, iters_s=1, additional_jump=2):
#         c, height, width = feat.shape
#         return TMPWrapper.TemporalMotionPropagation(feat,feat_pre,offsets,offsets_pre,distance,height,width,c,iters_t,sigma,iters_s,additional_jump)

# else: # only for test currently
#     def tmp(feat, feat_pre, offsets, offsets_pre, distance, iters_t=30, sigma=30, iters_s=1, additional_jump=2):
#         return None

def tmp(feat, feat_pre, offsets, offsets_pre, distance, iters_t=30, sigma=30, iters_s=1, additional_jump=2):
    raise NotImplementedError("Compilation of TMPWrapper to do")
    return None

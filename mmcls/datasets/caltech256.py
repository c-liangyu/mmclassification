# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path
import pickle

import numpy as np
import torch.distributed as dist

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class Caltech256(BaseDataset):
    CLASSES = list(range(257))

    def load_annotations(self):
        with open(self.ann_file, "r") as f:
            samples = [x.strip() for x in f.readlines()]

        data_infos = []
        for filename in samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(int(filename[:3]) - 1, dtype=np.int64)
            data_infos.append(info)

        return data_infos


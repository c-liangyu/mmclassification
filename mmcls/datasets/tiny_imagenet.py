# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path
import pickle

import numpy as np
import torch.distributed as dist

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class TinyImageNet(BaseDataset):
    CLASSES = list(range(200))

    def load_annotations(self):
        with open(self.ann_file, "r") as f:
            lines = [x.strip() for x in f.readlines()]

        data_infos = []
        for line in lines:
            prefix, filename, gt = line.split(" ")
            info = {'img_prefix': os.path.join(self.data_prefix, prefix)}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(int(gt), dtype=np.int64)
            data_infos.append(info)

        return data_infos


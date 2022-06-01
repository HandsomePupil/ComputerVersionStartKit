from __future__ import absolute_import
import sys
sys.path.append("/home/yg/CODE/VSCode/Tracking/siamfc-pytorch-master")

import os
from got10k.datasets import *


from siamfc import TrackerSiamFC


if __name__ == '__main__':
    # root_dir = os.path.expanduser('/home/yg/CODE/VSCode/Tracking/datasets/OTB')
    # seqs = OTB(root_dir, version=2015, download=False)
    root_dir = os.path.expanduser('/home/yg/CODE/VSCode/Tracking/datasets/GOT-10k')
    seqs = GOT10k(root_dir, subset='train', return_meta=True)
    tracker = TrackerSiamFC()
    tracker.train_over(seqs)

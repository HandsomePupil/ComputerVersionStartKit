from __future__ import absolute_import
import sys
sys.path.append("/home/yg/CODE/VSCode/Tracking/siamfc-pytorch-master")
import os
from got10k.experiments import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    net_path = 'pretrained/siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    # root_dir = os.path.expanduser('/home/yg/CODE/VSCode/Tracking/datasets/OTB')
    # e = ExperimentOTB(root_dir, version=2015)
    root_dir = os.path.expanduser('/home/yg/CODE/VSCode/Tracking/datasets/GOT-10k')
    e = ExperimentGOT10k(root_dir, subset='val')
    e.run(tracker,visualize=True)
    e.report([tracker.name])

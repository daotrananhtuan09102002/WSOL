import os
from trainer import Trainer
import numpy as np
import torch
import random

def set_random_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    set_random_seed(42)
    trainer = Trainer(dataset_name = 'CUB', architecture = 'resnet50', architecture_type = 'drop', pretrained = True,
                 large_feature_map = True, drop_threshold = 0.8, drop_prob = 0.25, lr = 0.002, lr_classifier_ratio = 10.0,
                 momentum = 0.9, weight_decay = 0.0001, lr_decay_points = [41, 61], lr_decay_rate = 0.2,
                 sim_fg_thres = 0.4, sim_bg_thres = 0.2, loss_ratio_drop = 2.0,
                 loss_ratio_sim = 0.5, loss_ratio_norm = 0.05, wsol_method = 'bridging-gap', loader = voc_dataloader, log_dir = '/conte')


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import os
import logging
import numpy as np
import torch

def seed_everywhere_torch(seed):
    """Initialize a random seed.
    From: https://github.com/pytorch/pytorch/issues/11278
    """
    assert type(seed) == int
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.tensor(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logging.info('Plant the random seed: {}.'.format(seed))


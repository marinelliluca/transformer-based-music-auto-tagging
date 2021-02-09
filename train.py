# coding: utf-8
# Heavily modified from https://github.com/minzwon/sota-music-tagging-models/
import os
import time
import numpy as np
from sklearn import metrics
import datetime
import tqdm
import pickle as pkl
import librosa
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import datetime


class Train(object):
    def __init__(self, main_dict)
    
    self.dataLoader = 
    
    # set up tensorboard
    now = datetime.datetime.now()
    log_dir = os.path.join("./","logs",
                           main_dict["dataset"],
                           main_dict["architecture"],
                           now.strftime("%m:%d:%H:%M"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"tensorboard --logdir '{log_dir}' --port ")
    self.writer = SummaryWriter(log_dir)
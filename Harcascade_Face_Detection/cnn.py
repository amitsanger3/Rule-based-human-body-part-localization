import cv2

import numpy as np
import matplotlib.pyplot as plt

import os, json
import random, traceback

import time

import shutil

import torch
from torch import nn
import torch.nn.functional as F


class NodeCNN(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=4, kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,)
        self.conv2 = nn.Conv2d(in_channels=4,out_channels=16, kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,)
        # self.conv2 = nn.Conv2d(in_channels=8,out_channels=16, kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,)
        
        self.fc1 = nn.Linear(2*2*16*16, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 2)
        
    def forward(self, X):
        
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2,2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2,2)
        # X = F.relu(self.conv3(X))
        # X = F.max_pool2d(X, 2,2)
        X = X.view(-1, 16*16*2*2)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        
        return F.log_softmax(X, dim=1)
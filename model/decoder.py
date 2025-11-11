import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils.utils import *
from model import nets


class Decoder(nn.Module):
    def __init__(self, dims= [], net= 'ToyNet'):
        super(Decoder, self).__init__()  
        self.net = getattr(nets, net)(dims)
    
    def forward(self, X):
        yHat = self.net(X)
        return yHat


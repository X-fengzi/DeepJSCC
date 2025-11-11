import os
import argparse
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
import copy
import torch.distributions as dist
import pickle
import math

class DataGen:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.call(dataset)

    def call(self, dataset= 'mnist'):
        """
        Load specified dataset.
        """
        if self.dataset == 'mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])#(0.1307,), (0.3081,)
            self.d_train = datasets.MNIST('./data/mnist/', train= True, download= True, transform= trans_mnist)
            self.d_test = datasets.MNIST('./data/mnist/', train= False, download= True, transform= trans_mnist)
            self.dim = (int(torch.prod(torch.tensor(self.d_train[0][0].shape))), 10)
    
        else: raise("Dataset is not well-preprocessed since no definition.")

    def batch(self, train= True, batch_size= 20, shuffle= True):
        return DataLoader(self.d_train, batch_size, shuffle) if train else DataLoader(self.d_test, batch_size)

    def get_data(self, x, y):
        batch_size = x.shape[0]
        x = x.view(batch_size,-1)   
        return x,y

    # customized functions for each dataset.

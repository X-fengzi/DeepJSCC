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
import logging

# import sionna as sn



def setup_logging(log_filename):
    logger = logging.getLogger(log_filename)
    format_str = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s')
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_filename)
    stream_handler.setFormatter(format_str)
    file_handler.setFormatter(format_str)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def IXZ_est(z, p):
    ps_list = list()
    for k in range(z.size(0)):
        ps_list.append(
            torch.roll(torch.exp(
                p.log_prob(torch.roll(z, shifts= -k, dims= 0))
            ), shifts= k, dims= 0)
        )
    ps = torch.stack(ps_list)
    log_pz = torch.log(torch.mean(ps, dim= 0))

    return torch.mean(torch.sum((p.log_prob(z) - log_pz), dim= 1))


def entropy_loss(probabilities, reduction='mean'):
    """
    计算概率分布的熵
    
    参数:
        probabilities (Tensor): shape (batch_size, num_classes)，必须是 softmax 后的概率
        reduction (str): 'mean' 或 'sum'，表示对 batch 维度如何聚合
    
    返回:
        Tensor: 熵 loss 值
    """
    # 防止 log(0) 导致数值不稳定
    eps = 1e-8
    probabilities = torch.clamp(probabilities, min=eps, max=1.0)

    # 计算熵 H(p) = -sum(p * log(p))
    entropies = -torch.sum(probabilities * torch.log(probabilities), dim=1)

    if reduction == 'mean':
        return entropies.mean()
    elif reduction == 'sum':
        return entropies.sum()
    else:
        return entropies  # 每个样本的熵

class WirelessChannel():
    '''
    1: Gaussian channel
    2: Rayleigh channel

    '''
    def __init__(self, channel_type, snr):
        self.channel_type = channel_type
        self.snr = snr
    def generate_ratio(self, data):
        if self.channel_type == 1 or self.channel_type == 'Gaussian':
            Gaussian_coefficient = torch.ones_like(data,requires_grad = False)
            ratio = Gaussian_coefficient
        elif self.channel_type == 2 or self.channel_type == 'Rayleigh':
            # 定义瑞利分布的尺度参数 sigma
            sigma = torch.sqrt(torch.tensor(2,requires_grad = False))/2  #1.0  # 可以根据需要调整这个值
            # 生成复数乘性衰落
            # ratio = torch.randn_like(data,requires_grad = False) * sigma
            ratio = torch.complex(torch.randn(1, device=data.device),torch.randn(1, device=data.device))* sigma   # Block Fading Channel
        else:
            raise RuntimeError("Channel is not right since no definition.")

        return ratio

    def transmit(self, data, snr=None):
        embd_dim = data.shape[-1]       
        data = torch.complex(data[...,:embd_dim//2],data[...,embd_dim//2:])
        ratio = self.generate_ratio(data)
        if snr:
            noise_ratio=10**(snr/10) 
        else:
            noise_ratio=10**(self.snr/10) 
        noise= torch.randn_like(data,requires_grad = False)/noise_ratio #/torch.sqrt(torch.tensor(data.shape[1]))
        Rx_sig = torch.mul(ratio,data)+noise
        return torch.cat([Rx_sig.real,Rx_sig.imag],dim=-1)

class BinarySymmetricChannel:
    def __init__(self, p):
        """
        Args:
            p (float): bit error rate, 0 <= p <= 1
        """
        assert 0 <= p <= 1, "p must be in [0, 1]"
        self.p = p

    def transmit(self, z):
        """
        Args:
            z: torch.Tensor, shape [batch_size, length], values in {0, 1}, dtype=torch.long or torch.int
        Returns:
            received: same shape and dtype as z, with bits flipped independently with probability p
        """
        if not (z.dtype == torch.long or z.dtype == torch.int or z.dtype == torch.bool):
            raise ValueError("Input z must be integer-type (0/1), e.g., torch.long")

        # 生成翻转掩码：True 表示该位要翻转
        flip_mask = torch.rand_like(z, dtype=torch.float,device=z.device) < self.p  # [B, L], bool

        # 翻转比特：0->1, 1->0
        received = z ^ flip_mask.long()  # XOR with 0/1 mask

        return received

def one_hot_to_bits(one_hot: torch.Tensor) -> torch.Tensor:
    """
    将 shape 为 (batch, n, k) 的 one-hot 张量转换为 (batch, n * bits_per_symbol) 的比特流。
    """
    batch, n, k = one_hot.shape
    bits_per_symbol = math.ceil(math.log2(k))
    
    # 找到 one-hot 中为 1 的位置（即类别索引）
    indices = torch.argmax(one_hot, dim=-1)  # shape: (batch, n)

    # 转换为二进制表示
    # 使用位运算：逐位提取
    masks = 1 << torch.arange(bits_per_symbol - 1, -1, -1, device=one_hot.device)  # shape: (bits_per_symbol,)
    bits = (indices.unsqueeze(-1) & masks) != 0  # shape: (batch, n, bits_per_symbol)
    bits = bits.long()  # 转为 0/1 float

    # 展平为 (batch, n * bits_per_symbol)
    bits = bits.view(batch, -1)
    return bits

def bits_to_one_hot(bits: torch.Tensor, k: int) -> torch.Tensor:
    """
    将 shape 为 (batch, n * bits_per_symbol) 的比特流还原为 (batch, n, k) 的 one-hot 张量。
    """
    batch, total_bits = bits.shape
    bits_per_symbol = math.ceil(math.log2(k))
    assert total_bits % bits_per_symbol == 0, "总比特数必须能被 bits_per_symbol 整除"
    n = total_bits // bits_per_symbol

    # 重塑为 (batch, n, bits_per_symbol)
    bits = bits.view(batch, n, bits_per_symbol)

    # 转回整数索引
    powers = 2 ** torch.arange(bits_per_symbol - 1, -1, -1, device=bits.device)  # shape: (bits_per_symbol,)
    indices = (bits * powers).sum(dim=-1).long()  # shape: (batch, n)

    # 处理非法索引（超出 [0, k-1]）
    indices = torch.clamp(indices, 0, k - 1)

    # 转为 one-hot
    one_hot = torch.zeros(batch, n, k, device=bits.device)
    one_hot.scatter_(2, indices.unsqueeze(-1), 1.0)
    return one_hot

def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config.
    """
    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object.
            if x.lower() == "none": return None
            # If default is None (and x is not None), return x without conversion as str.
            elif default is None: return str(x)
            # Otherwise, default has non-None type; convert x to that type.
            else: return type(default)(x)
        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument.
                    parser.add_argument(
                        f"--{param}",
                        action="append",
                        type=type(default[0]),
                        default=default,
                        help=description
                    )
                else: parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else: parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser

if __name__ == "__main__":
    pass

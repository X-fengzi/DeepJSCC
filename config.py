import argparse
from utils.utils import add_flags_from_config
import yaml
import json

config_args = {
    'method_config': {
        'method': ('EIB', 'which method to use (DeepJSCC, DT_Gumbel)'),
    },
    'data_config': {
        'dataset': ('mnist', 'which dataset to use'), 
    },
    'model_config': {
        'task': ('classification', 'which tasks to train on, can be any of [classification, inference]'),
        'encoder-model': ('LinearModel', 'which encoder to use'),
        'decoder-model': ('ToyNet', 'which decoder to use'),
        'embd-dim': (96, 'embedding dimension'),
    },
    'training_config': {
        'lr': (1e-4, 'learning rate'),
        'beta': (1e-4, 'bottleneck coefficient'),
        'bs': (30, 'batch size'),
        'cuda': (3, 'cuda device (-1 for cpu training)'),
        'epochs': (100, 'number of training epochs'),
        'PSNR': (True, 'if select PSNR as channel parameter else SNR'), #Analog Communication
        'BER':(0.1,'Bit error rate'), #Digital Communication
        'seed': (3407, 'seed for training'),
        'log-freq': (100, 'printing frequency of train/val metrics (in batchs)'),
        'grad-clip': (50, 'max norm for gradient clipping, or None for no gradient clipping'),
    },
}


def parse_sub_method(s):
    return list(s.split(','))


# load YAML config
def load_yaml_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_args():
    parser = argparse.ArgumentParser()
    for _, config_dict in config_args.items():
        parser = add_flags_from_config(parser, config_dict)
        
    parser.add_argument('--config_file_path', type=str, default=None, help='Path to the YAML config file')
    
    ############# set sub method(scene) ##################
    parser.add_argument('--sub_method', type=parse_sub_method,default='',help='sub method or scenario (IB,SC)')
    # IB: Infomation bottleneck as the loss function
    # SC: source coding (without channe;)

    args = parser.parse_args()

    if args.config_file_path:
        config_dict = load_yaml_config(args.config_file_path)
        
        # update args
        for key, value in config_dict.items():
            if hasattr(args, key):
                setattr(args, key, value)
    return args
    
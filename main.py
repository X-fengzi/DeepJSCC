import torch
import torch.nn.functional as F
import math

from datetime import datetime

from model import decoder, encoder
from utils.utils import *
from datasets import *
import config
import json
from function import *

# ----------------- pre-process -----------------
# load configs.
args = config.get_args()
set_seed(args.seed)

# set log
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = './logs/' + f'{args.dataset}'+'/'+ f'{args.method}' + '/'+f'{current_time}.log' #+ args.task
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logger = setup_logging(log_filename)

# set training config
device = torch.device(f"cuda:{args.cuda}" if (torch.cuda.is_available()) else "cpu") #args.cuda and
print("Using device:", device)
# load datasets & additional arguements.
data = DataGen(args.dataset)
args.input_dim, args.output_dim = data.dim

# --------------------------------------------------  DeepJSCC   ------------------------------------------------------

if args.method == 'DeepJSCC':
    # initialize en/decoders.

    Encoder = encoder.GaussianEncoder(dims=[args.input_dim, 2*int(args.embd_dim)], net=args.encoder_model)
    Decoder = decoder.Decoder(dims=[int(args.embd_dim), args.output_dim], net=args.decoder_model)

    Encoder.to(device)
    Decoder.to(device)
    # define optimizer.
    params = []
    for param in [list(Encoder.parameters())] + [list(Decoder.parameters())]:
        params += param
    optimizer = torch.optim.Adam(params, lr= args.lr)
    if('SC' in args.sub_method):
        channel = None
    else:
        channel = WirelessChannel(1,0)

    final_config = vars(args) 
    with open(log_filename, 'w') as f:
        json.dump(final_config, f, indent=4)
            # logger.info(f"Final configuration saved to {log_file}")
        f.write("\n")

    # initial test
    logger.info("----------------- initial test -----------------")

    test_DeepJSCC(0, data, Encoder, Decoder, channel, device, args, logger)

    # ----------------- train ----------------- 
    logger.info("----------------- train start... -----------------")

    for epoch in range(1,args.epochs+1):
        train_DeepJSCC(epoch, data, Encoder, Decoder, channel, optimizer, params, device, args, logger)
        test_DeepJSCC(epoch, data, Encoder, Decoder, channel, device, args, logger)

# --------------------------------------------------  DiscreteDeepJSCC   ------------------------------------------------------

if args.method == 'DT_VQVAE':
    # initialize en/decoders.
    Encoder = encoder.BinaryVQEncoder(dims=[args.input_dim, int(args.embd_dim)], net=args.encoder_model)
    Decoder = decoder.Decoder(dims=[int(args.embd_dim), args.output_dim], net=args.decoder_model)

    Encoder.to(device)
    Decoder.to(device)
    # define optimizer.
    params = []
    for param in [list(Encoder.parameters())] + [list(Decoder.parameters())]:
        params += param

    optimizer = torch.optim.Adam(params, lr= args.lr)

    if('SC' in args.sub_method):
        channel = None
    else:
        channel = BinarySymmetricChannel(args.BER)

    final_config = vars(args)  
    with open(log_filename, 'w') as f:
        json.dump(final_config, f, indent=4)
            # logger.info(f"Final configuration saved to {log_file}")
        f.write("\n")

    # initial test
    logger.info("----------------- initial test -----------------")

    test_DT_VQVAE(0, data, Encoder, Decoder, channel, device, args, logger)

    # ----------------- train ----------------- 
    logger.info("----------------- train start... -----------------")

    for epoch in range(1,args.epochs+1):
        train_DT_VQVAE(epoch, data, Encoder, Decoder, channel, optimizer, params, device, args, logger)
        test_DT_VQVAE(epoch, data, Encoder, Decoder, channel, device, args, logger)

    

if args.method == 'DT_Gumbel':
    # initialize en/decoders.
    Encoder = encoder.DiscreteEncoder(dims=[args.input_dim, int(args.embd_dim)], net=args.encoder_model)
    Decoder = decoder.Decoder(dims=[int(args.embd_dim), args.output_dim], net=args.decoder_model)

    Encoder.to(device)
    Decoder.to(device)
    # define optimizer.
    params = []
    for param in [list(Encoder.parameters())] + [list(Decoder.parameters())]:
        params += param

    optimizer = torch.optim.Adam(params, lr= args.lr)

    if('SC' in args.sub_method):
        channel = None
    else:
        channel = BinarySymmetricChannel(args.BER)

    final_config = vars(args) 
    with open(log_filename, 'w') as f:
        json.dump(final_config, f, indent=4)
            # logger.info(f"Final configuration saved to {log_file}")
        f.write("\n")

    # initial test
    logger.info("----------------- initial test -----------------")

    test_DT_Gumbel(0, data, Encoder, Decoder, channel, device, args, logger)

    # ----------------- train ----------------- 
    logger.info("----------------- train start... -----------------")

    for epoch in range(1,args.epochs+1):
        train_DT_Gumbel(epoch, data, Encoder, Decoder, channel, optimizer, params, device, args, logger)
        test_DT_Gumbel(epoch, data, Encoder, Decoder, channel, device, args, logger)














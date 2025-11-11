import torch
import torch.nn.functional as F
import logging
import random
from utils.utils import *


def train_DeepJSCC(epoch, data, Encoder, Decoder, channel, optimizer, params, device, args, logger):
    """
    train one epoch
    """
    Encoder.train()
    Decoder.train()
    count = 0
    for x, y in data.batch(batch_size= args.bs):
        x, y = data.get_data(x,y)
        x, y = x.to(device),  y.to(device)
            # initialize common-repr z.
        IclubXZ = 0
        z, qzxk = Encoder(x)
        IclubXZ = IXZ_est(z,qzxk)
        # update z for any modality or task.
        if args.PSNR:
            z_norm = torch.tanh(z)
        else:
            z_norm = z/torch.norm(z, p=2, dim=1, keepdim=True)*torch.sqrt(torch.tensor(z.shape[1]))
        # 信道
        if(channel):
            zHat = channel.transmit(z_norm)
        else:
            zHat = z_norm
        InceZY = 0
        optimizer.zero_grad() 
        loss = 0
        InceZY = -F.cross_entropy(Decoder(zHat), y) + torch.log(torch.tensor(10))
        loss = -InceZY
        if("IB" in args.sub_method):
            loss+=args.beta*IclubXZ
        # optimize qzx & qyz. 
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(parameters = params, max_norm= args.grad_clip)
        # nat -> bit
        IclubXZ = IclubXZ/torch.log(torch.tensor(2))
        InceZY = InceZY/torch.log(torch.tensor(2))
        # console outputs.
        if count % args.log_freq == 0: 
            logger.info(f"epoch-batch: {epoch}-{count} iter loss: {loss.item()}, train_IXZ: {IclubXZ.item()}, train_IZY: {InceZY.item()}")
        count += 1
    logger.info(f"epoch-batch: {epoch}-{count} iter loss: {loss.item()}, train_IXZ: {IclubXZ.item()}, train_IZY: {InceZY.item()}")


def test_DeepJSCC(epoch, data, Encoder, Decoder, channel, device, args, logger):
    Encoder.eval()
    Decoder.eval()
    IclubXZ, InceZY, acc_num, num = 0, 0, 0, 0
    for x, y in data.batch(batch_size= args.bs, train=False):
        x, y = data.get_data(x,y)
        x, y = x.to(device),  y.to(device)
        # compute IXZ via club. 
        z, qzxk = Encoder(x)
        IclubXZ += IXZ_est(z,qzxk)
        # update z for any modality or task.
        if args.PSNR:
            z_norm = torch.tanh(z)
        else:
            z_norm = z/torch.norm(z, p=2, dim=1, keepdim=True)*torch.sqrt(torch.tensor(z.shape[1]))
        if(channel):
            zHat = channel.transmit(z_norm)
        else:
            zHat = z_norm
        InceZY+=-F.cross_entropy(Decoder(zHat), y) \
            + torch.log(torch.tensor(10))
        _, predicted = torch.max(Decoder(zHat), 1)
        acc_num += (predicted == y).sum() 
        num += y.shape[0]
    IclubXZ = IclubXZ*(args.bs/num)
    InceZY = InceZY*(args.bs/num)
    distortion = torch.log(torch.tensor(10)) - InceZY
    # nat -> bit
    IclubXZ = IclubXZ/torch.log(torch.tensor(2))
    InceZY = InceZY/torch.log(torch.tensor(2))
    acc = acc_num/num
    if epoch:
        logger.info(
        f"epoch {epoch}  test_IXZ: {IclubXZ.item()}, test_IZY: {InceZY.item()}, distortion: {distortion}, ACC:{acc}"
        )
    else:
        logger.info(
            f"start:  test_IXZ: {IclubXZ.item()}, test_IZY: {InceZY.item()}, distortion: {distortion}, ACC:{acc}"
        )


def train_DT_VQVAE(epoch, data, Encoder, Decoder, channel, optimizer, params, device, args, logger):
    """
    train one epoch
    """
    Encoder.train()
    Decoder.train()
    count = 0
    for x, y in data.batch(batch_size= args.bs):
        x, y = data.get_data(x,y)
        x, y = x.to(device),  y.to(device)
        # compute IXZ via club.
        vq_loss = 0
        z_bits, codebooks, quantized, z_e = Encoder(x)
        vq_loss = F.mse_loss(quantized, z_e.detach()) + 0.25*F.mse_loss(z_e, quantized.detach())
        if(channel):
            zHat_bits = channel.transmit(z_bits)
        else:
            zHat_bits = z_bits
        zHat = quantized + Encoder.reconstruct_from_bits(zHat_bits) - quantized.detach()
        # update z for any modality or task.
        InceZY = 0
        optimizer.zero_grad() 
        loss = 0
        ce_loss = F.cross_entropy(Decoder(zHat), y)
        loss = ce_loss + args.beta*vq_loss
        InceZY = -ce_loss + torch.log(torch.tensor(10))
        # optimize qzx & qyz. 
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(parameters = params, max_norm= args.grad_clip)
        # nat -> bit
        InceZY = InceZY/torch.log(torch.tensor(2))
        # console outputs.
        if count % args.log_freq == 0: 
            logger.info(f"epoch-batch: {epoch}-{count} iter loss: {loss.item()}, train_IZY: {InceZY.item()}")
        count += 1
    logger.info(f"epoch-batch: {epoch}-{count} iter loss: {loss.item()}, train_IZY: {InceZY.item()}")

def test_DT_VQVAE(epoch, data, Encoder, Decoder, channel, device, args, logger):
    Encoder.eval()
    Decoder.eval()
    InceZY, acc_num, num = 0, 0, 0
    for x, y in data.batch(batch_size= args.bs, train=False):
        x, y = data.get_data(x,y)
        x, y = x.to(device),  y.to(device)
        z_bits, codebooks, quantized, z_e = Encoder(x)
            # 信道
        if(channel):
            zHat_bits = channel.transmit(z_bits)
        else:
            zHat_bits = z_bits
        zHat = Encoder.reconstruct_from_bits(zHat_bits)
        InceZY+=-F.cross_entropy(Decoder(zHat), y) + torch.log(torch.tensor(10))
        _, predicted = torch.max(Decoder(zHat), 1)
        acc_num += (predicted == y).sum() 
        num += y.shape[0]
    InceZY = InceZY*(args.bs/num)
    distortion = torch.log(torch.tensor(10)) - InceZY
    # nat -> bit
    InceZY = InceZY/torch.log(torch.tensor(2))
    acc = acc_num/num
    if epoch:
        logger.info(
        f"epoch {epoch}  rate: {args.embd_dim}, distortion: {distortion}, test_IZY: {InceZY.item()}, ACC:{acc}"
        )
    else:
        logger.info(
            f"start:  rate: {args.embd_dim}, distortion: {distortion}, test_IZY: {InceZY.item()}, ACC:{acc}"
        )

def train_DT_Gumbel(epoch, data, Encoder, Decoder, channel, optimizer, params, device, args, logger):
    """
    train one epoch
    """
    Encoder.train()
    Decoder.train()
    count = 0
    for x, y in data.batch(batch_size= args.bs):
        x, y = data.get_data(x,y)
        x, y = x.to(device),  y.to(device)
        IclubXZ = 0
        # compute IXZ via club.
        z, codebook, dist = Encoder(x)
        z_index = torch.argmax(z, dim=-1)
        IclubXZ += torch.mean(
            dist.log_prob(z_index) \
                - dist.log_prob(z_index[torch.randperm(z_index.size(0)), :])
        )
        z_bits = one_hot_to_bits(z)
        if(channel):
            zHat_bits = channel.transmit(z_bits)
        else:
            zHat_bits = z_bits
        zHat_one_hot = z + bits_to_one_hot(zHat_bits,z.shape[2]) - F.one_hot(torch.argmax(z,dim=-1),num_classes=z.shape[2])
        zHat = torch.matmul(zHat_one_hot,codebook).view(-1,args.embd_dim)
            
        # update z for any modality or task.
        InceZY = 0
        optimizer.zero_grad() 
        loss = 0
        ce_loss = F.cross_entropy(Decoder(zHat), y)
        loss = ce_loss # + Ixz
        InceZY = -ce_loss + torch.log(torch.tensor(10))
        # optimize qzx & qyz. 
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(parameters = params, max_norm= args.grad_clip)
        # nat -> bit
        InceZY = InceZY/torch.log(torch.tensor(2))
        # console outputs.
        if count % args.log_freq == 0: 
            logger.info(f"epoch-batch: {epoch}-{count} iter loss: {loss.item()}, train_IZY: {InceZY.item()}")
        count += 1
    logger.info(f"epoch-batch: {epoch}-{count} iter loss: {loss.item()}, train_IZY: {InceZY.item()}")

def test_DT_Gumbel(epoch, data, Encoder, Decoder, channel, device, args, logger):
    Encoder.eval()
    Decoder.eval()
    IclubXZ, InceZY, acc_num, num = 0, 0, 0, 0
    for x, y in data.batch(batch_size= args.bs, train=False):
        x, y = data.get_data(x,y)
        x, y = x.to(device),  y.to(device)
        IclubXZ = 0
        z, codebook, dist = Encoder(x)
        z_index = torch.argmax(z, dim=-1)
        IclubXZ += torch.mean(
            dist.log_prob(z_index) \
                - dist.log_prob(z_index[torch.randperm(z_index.size(0)), :])
        )
        # onehot convert to bit sequence
        z_bits = one_hot_to_bits(z)
        if(channel):
            zHat_bits = channel.transmit(z_bits)
        else:
            zHat_bits = z_bits
        zHat_one_hot = bits_to_one_hot(zHat_bits,z.shape[2])
        zHat = torch.matmul(zHat_one_hot,codebook).view(-1,args.embd_dim)
        # update z for any modality or task.
        InceZY+=-F.cross_entropy(Decoder(zHat), y) + torch.log(torch.tensor(10))
        _, predicted = torch.max(Decoder(zHat), 1)
        acc_num += (predicted == y).sum() 
        num += y.shape[0]
    InceZY = InceZY*(args.bs/num)
    distortion = torch.log(torch.tensor(10)) - InceZY
    # nat -> bit
    InceZY = InceZY/torch.log(torch.tensor(2))
    acc = acc_num/num
    if epoch:
        logger.info(
        f"epoch {epoch}  rate: {args.embd_dim}, distortion: {distortion}, test_IZY: {InceZY.item()}, ACC:{acc}"
        )
    else:
        logger.info(
            f"start:  rate: {args.embd_dim}, distortion: {distortion}, test_IZY: {InceZY.item()}, ACC:{acc}"
        )

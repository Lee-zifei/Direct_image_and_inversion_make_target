import os
import time
import argparse
import datetime
import numpy as np
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
from losses import L1
from metrics import AverageMeter,snr
from config import get_config
from models import build_model
from data.build_iter_dataset import build_load_folder as build_loader
from data.build_iter_dataset import patching, patch_single
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", default='configs/DT2_6.yaml',help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, default=16,help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')
    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config

def main(config,pre_train=False):
    dataset_train, dataset_val, data_loader_train, data_loader_val= build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)

    if pre_train:
        model_dict = model.state_dict()
        pretrained_dict2 = torch.load("WUDTnet.pth", map_location='cpu')['model']
        new_state_dict = OrderedDict()
        for k, v in model_dict.items():
            name = k
            new_state_dict[name] = v
        for k, v in pretrained_dict2.items():
            if "denoise" not in  k:
                new_state_dict[k]=v
        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict, strict=True)
        for name, param in model.named_parameters():
            if "denoise" not in name:
                param.requires_grad = False

    # logger.info(str(model))
    optimizer = build_optimizer(config, model)
    if len(config.GPU) > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    if len(config.GPU) > 1:
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    max_accuracy = 0.0
    min_loss = 1.0
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
    if config.MODEL.RESUME:
        max_accuracy, min_loss = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        if config.TRAIN.LOSS.NAME == 'MSE':
            criterion = torch.nn.MSELoss()# MIX3()
        elif config.DENOSIE:
            criterion = L1()

        acc1, loss = validate(config, data_loader_val, model,criterion)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        if config.TRAIN.LOSS.NAME == 'MSE':
            criterion = torch.nn.MSELoss()#MIX3()
        if 'NAFnet' == config.MODEL.TYPE:
            criterion = L1()
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler)
        if  (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)) and 'preprocess'not in config.MODEL.NAME:
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, min_loss, optimizer, lr_scheduler, logger)

        acc1, loss = validate(config, data_loader_val, model,criterion)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss:.8f}")
        min_loss = min(min_loss, loss)
        if min_loss==loss  and 'preprocess'not in config.MODEL.NAME:
            torch.save(model.state_dict(), f'{config.MODEL.NAME}.pth')
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max snr: {max_accuracy:.3f}')
        logger.info(f'Min loss: {min_loss:.8f}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler):

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    snr_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (images, labels ,names) in enumerate(data_loader):

        samples, targets, index = patching(config,images,names[0],targets=labels,denoise=config.DENOISE)
        if config.DENOISE:
            index = index.cuda()
        samples = samples.cuda()
        targets = targets.cuda()

        output = model(samples)
        if config.DENOISE:
            output = model(samples, index)
        loss = criterion(output, targets)
        snr1 =  snr(output, targets)

        model.train()
        optimizer.zero_grad()
        loss.backward()
        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        else:
            grad_norm = get_grad_norm(model.parameters())
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)

        snr_meter.update(snr1.item(), targets.size(0))
        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.8f} ({loss_meter.avg:.8f})\t'
                f'snr {snr_meter.val:.4f} ({snr_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})'
            )
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model,criterion):

    model.eval()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()

    end = time.time()
    for idx, (images, labels, names) in enumerate(data_loader):
        samples, targets, index = patching(config, images, names[0], targets=labels, test=True, denoise=config.DENOISE)
        if config.DENOISE:
            index = index.cuda(non_blocking=True)
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        output = model(samples)
        if config.DENOISE:
            output = model(samples, index)
        loss = criterion(output, targets)
        acc1 =  snr(output, targets)

        loss_meter.update(loss.item(), targets.size(0))
        acc1_meter.update(acc1.item(), targets.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss:  {loss_meter.val:.8f} ({loss_meter.avg:.8f})\t'
                f'snr:  {acc1_meter.val:.4f} ({acc1_meter.avg:.4f})')
    logger.info(f' * snr: {acc1_meter.avg:.3f} ')
    return acc1_meter.avg, loss_meter.avg


if __name__ == '__main__':
    _, config = parse_option()
    # if len(config.GPU)>1:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in config.GPU)
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    pretrain=False
    main(config,pretrain)


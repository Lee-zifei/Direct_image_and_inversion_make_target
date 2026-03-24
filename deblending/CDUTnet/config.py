import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']
_C.WAMDB = 'False'
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 64
# Input image size
_C.DATA.IMG_SIZE = 64
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 1
_C.GPU = [4,5,0]
_C.DATA.DATA_PATH='data/NewData3/Train'
_C.DATA.TEST_PATH='data/NewData3/Test'
_C.DATA.TEST_TYPE='field'
# _C.DATA.NT=1024
# _C.DATA.NX=32

_C.DATA.ROW=1024
_C.DATA.SROW=32
_C.DATA.COL=512
_C.DATA.SCOL=35
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'dt'
# Model name
_C.MODEL.NAME = 'DTv2'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1
# Dropout rate
_C.MODEL.DROP_RATE = 0.1
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1  ### need change to 0.1

# Swin Transformer parameters
_C.MODEL.DT = CN()
_C.MODEL.DT.IN_CHANS = 1
_C.MODEL.DT.EMBED_DIM = 64
_C.MODEL.DT.DEPTHS = [2, 2, 2, 2]
_C.MODEL.DT.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.DT.WINDOW_SIZE = [8,16,32,64]
_C.MODEL.DT.MLP_RATIO = 4.
_C.MODEL.DT.QKV_BIAS = True
_C.MODEL.DT.QK_SCALE = 0
_C.MODEL.DT.PATCH_NORM = True
_C.TESTF=False
_C.VAL=False
_C.DENOISE=False
# 
# # Deblending Transformer 2 parameters
_C.MODEL.DT2 = CN()
_C.MODEL.DT2.IN_CHANS = 1
_C.MODEL.DT2.EMBED_DIM =  64
_C.MODEL.DT2.DEPTHS = [4, 7, 19, 8]
_C.MODEL.DT2.NUM_HEADS = [2, 3, 7, 10]
_C.MODEL.DT2.NITER = [1, 1, 1, 1]
_C.MODEL.DT2.STOKEN_SIZE = [8, 4, 1, 1]
_C.MODEL.DT2.MLP_RATIO = 4.
_C.MODEL.DT2.QKV_BIAS = True
_C.MODEL.DT2.QK_SCALE = 0
_C.MODEL.DT2.PATCH_NORM = True
#-----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 100
_C.TRAIN.WARMUP_EPOCHS = 4 ## need change to 10
_C.TRAIN.WEIGHT_DECAY = 1e-4 ## need change to 0.01
_C.TRAIN.BASE_LR = 2e-4 #1e-3
_C.TRAIN.WARMUP_LR = 1e-4  #1e-4
_C.TRAIN.MIN_LR = 1e-6 #1e-8
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
_C.TRAIN.GID = (2,3,4,5)
# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.99)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
# Loss
_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.NAME = 'MSE'
# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.TYPE = 'test3'
_C.TEST.MODE = False
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = 'O0'
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 1
# Fixed random seed
_C.SEED = 3207   ### or the university answer 42
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
_C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.local_rank:
        config.LOCAL_RANK = args.local_rank
    else:
        config.LOCAL_RANK = 0
    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config

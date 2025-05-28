def update_config(config, yaml_file):
    from easydict import EasyDict as edict
    import yaml
    with open(yaml_file) as f:
        yaml_config = edict(yaml.load(f, Loader=yaml.SafeLoader))
        for k, v in yaml_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        if vk in config[k]:
                            config[k][vk] = vv
                else:
                    config[k] = v
    return config

def get_config(weights_path = ''):
    from easydict import EasyDict as edict
    import os

    cfg = edict()
    # Model configuration
    cfg.MODEL = edict()
    cfg.MODEL.NAME = 'pose_hrnet'
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.INIT_WEIGHTS = True
    cfg.MODEL.PRETRAINED = os.path.abspath(weights_path)
    cfg.MODEL.NUM_JOINTS = 17  # Number of joints (17 for COCO)
    cfg.MODEL.IMAGE_SIZE = [384, 288]  # Input size [width, height]
    cfg.MODEL.EXTRA = edict()
    cfg.MODEL.EXTRA.PRETRAINED_LAYERS = ['*']
    cfg.MODEL.EXTRA.FINAL_CONV_KERNEL = 1

    # Stage 2 configuration
    cfg.MODEL.EXTRA.STAGE2 = edict()
    cfg.MODEL.EXTRA.STAGE2.NUM_MODULES = 1
    cfg.MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
    cfg.MODEL.EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
    cfg.MODEL.EXTRA.STAGE2.NUM_CHANNELS = [32, 64]
    cfg.MODEL.EXTRA.STAGE2.BLOCK = 'BASIC'
    cfg.MODEL.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

    # Stage 3 configuration
    cfg.MODEL.EXTRA.STAGE3 = edict()
    cfg.MODEL.EXTRA.STAGE3.NUM_MODULES = 4
    cfg.MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
    cfg.MODEL.EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
    cfg.MODEL.EXTRA.STAGE3.NUM_CHANNELS = [32, 64, 128]
    cfg.MODEL.EXTRA.STAGE3.BLOCK = 'BASIC'
    cfg.MODEL.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

    # Stage 4 configuration
    cfg.MODEL.EXTRA.STAGE4 = edict()
    cfg.MODEL.EXTRA.STAGE4.NUM_MODULES = 3
    cfg.MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
    cfg.MODEL.EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
    cfg.MODEL.EXTRA.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
    cfg.MODEL.EXTRA.STAGE4.BLOCK = 'BASIC'
    cfg.MODEL.EXTRA.STAGE4.FUSE_METHOD = 'SUM'

    # Testing configuration
    cfg.TEST = edict()
    cfg.TEST.BATCH_SIZE = 32
    cfg.TEST.POST_PROCESS = True
    cfg.TEST.SHIFT_HEATMAP = True

    return cfg

def update_config(cfg, dataset_dir, batchsize = 4, num_epochs=20):
    from copy import deepcopy
    from easydict import EasyDict as edict
    
    # Create a modified config for finetuning
    finetune_cfg = deepcopy(cfg)
    
    finetune_cfg.DATASET = edict()
    finetune_cfg.DATASET.ROOT = dataset_dir
    finetune_cfg.DATASET.TRAIN_SET = 'annotations/train.json'
    finetune_cfg.DATASET.TEST_SET = 'annotations/val.json'
    
    # Finetuning hyperparameters
    finetune_cfg.TRAIN = edict()
    finetune_cfg.TRAIN.BEGIN_EPOCH = 0
    finetune_cfg.TRAIN.END_EPOCH = num_epochs  # Fewer epochs for finetuning
    finetune_cfg.TRAIN.BATCH_SIZE_PER_GPU = batchsize
    finetune_cfg.TRAIN.LR = 1e-5  # Much lower learning rate
    finetune_cfg.TRAIN.LR_STEP = [10, 15]
    finetune_cfg.TRAIN.LR_FACTOR = 0.1
    
    # Output directories
    finetune_cfg.OUTPUT_DIR = 'finetune_output'
    finetune_cfg.LOG_DIR = 'finetune_logs'
    
    return finetune_cfg
from dataclasses import dataclass
import os
import pandas as pd
from utils import get_default_device
from model import generate_samples


@dataclass
class BaseConfig:
    def __init__(self, dataset):
        self.DEVICE = get_default_device()
        self.DATASET = dataset
        # For logging inferece images and saving checkpoints.
        self.root_log_dir = os.path.join("Logs_Checkpoints", "Inference")
        self.root_checkpoint_dir = os.path.join("Logs_Checkpoints", "checkpoints")
        # Current log and checkpoint directory.
        self.log_dir = "version_0"
        self.checkpoint_dir = "version_0"

@dataclass
class TrainingConfig:
    def __init__(self, tsteps, epochs, bsize, lr):
        self.TIMESTEPS = tsteps # Define number of diffusion timesteps
        self.IMG_SHAPE = (1, 104) #(1, 32, 32) if BaseConfig.DATASET == "MNIST" else (3, 32, 32)  # (Channels, Height, Width) -> for our implementation would be (1, 1, 256)
        self.NUM_EPOCHS = epochs #30
        self.BATCH_SIZE = bsize#128
        self.LR = lr #2e-4
        self.NUM_WORKERS = 2

@dataclass
class ModelConfig:
    def __init__(self, base_ch, base_ch_mult, att, drop, time_mult):
        self.BASE_CH = base_ch#64  # 64, 128, 256, 512
        self.BASE_CH_MULT = base_ch_mult # 32, 16, 8, 4
        self.APPLY_ATTENTION = att
        self.DROPOUT_RATE = drop #0.01 #0.1
        self.TIME_EMB_MULT = time_mult # 128

def create_base_config(dataset = 'zbot'):
    return BaseConfig(dataset=dataset)

def create_train_config(timesteps = 1000, num_epochs = 50, batch_size = 32, lr = 0.00005):
    return TrainingConfig(tsteps=timesteps, epochs=num_epochs, bsize=batch_size, lr=lr)

def create_model_config(base_ch = 32, base_ch_mult = (1,2,4,8), att = (False, False, False, False), drop = 0.01, time_mult = 2):
    return ModelConfig(base_ch=base_ch, base_ch_mult=base_ch_mult, att=att, drop=drop, time_mult=time_mult)


if __name__ == '__main__':
    # Hyperparameters
    lr = 5e-05
    b_ch = 32
    att = (True, True, True, True)
    t = 2
    # Configurations
    base = create_base_config(dataset='Injector')
    train = create_train_config(num_epochs=150, lr=lr)
    model = create_model_config(base_ch=b_ch, att=att, time_mult=t)

    # Get Samples
    generate_samples(base, model, train, 
        log_folder = "diffusion\Logs_Checkpoints",
        version = 64, num_samples = 2000)
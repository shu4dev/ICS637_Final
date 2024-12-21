#Crdit: https://www.kaggle.com/code/voix97/jane-street-rmf-training-nn
import os
import pickle
import polars as pl
import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer
from pytorch_lightning.loggers import WandbLogger
import wandb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import gc

class custom_args():
    def __init__(self):
        self.usegpu = True
        self.gpuid = 0
        self.seed = 42
        self.model = 'AE'
        self.patience = 5
        self.max_epochs = 50
        self.N_fold = 5
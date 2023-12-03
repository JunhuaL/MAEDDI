import os
import warnings
import numpy as np
import yaml
import pandas as pd
import yaml
import argparse
import torch as t 
from torch import Tensor
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import Trainer

from PreModel import SMILEMAE 
from dataset import SMILES_DataModule
from utils import * 


if __name__ == '__main__':
    n_epochs = 80
    save_folder = './dataset/DrugBank_pretraining/drug/cleaned_data.csv'
    datamodule = SMILES_DataModule(save_folder,64,'drug',split_strat='whole_cluster_sampling')

    model = SMILEMAE(67,128)

    earlystopping_tracking = 'val_loss'
    earlystopping_mode = 'min'
    earlystopping_min_delta = 0.0001

    save_model_folder = f'./model_checkpoints/cnn_max_epoch_{n_epochs}_one_cls/'

    checkpoint_callback = pl_callbacks.ModelCheckpoint(dirpath=save_model_folder,
                                        mode = earlystopping_mode,
                                        monitor=earlystopping_tracking,
                                        save_top_k=1,save_last=True,)
    earlystop_callback = pl_callbacks.EarlyStopping(earlystopping_tracking,verbose=True,
                                        mode = earlystopping_mode,
                                        min_delta=earlystopping_min_delta,
                                        patience=10,)
    
    trainer = Trainer(
                    gpus=[0,],
                    accelerator=None,
                    max_epochs=n_epochs, min_epochs=5,
                    default_root_dir= save_model_folder,
                    fast_dev_run=False,
                    check_val_every_n_epoch=1,
                    callbacks=[checkpoint_callback,]
                        #     earlystop_callback,],
                    )
    trainer.fit(model, datamodule=datamodule,)
import sys
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import Trainer

from MolConfSSL import PreModel_Container 
from large_dataset import Large_PretrainingDataset
from utils import * 


if __name__ == '__main__':
    n_epochs = int(sys.argv[1])
    n_layers = int(sys.argv[2])
    split_strat = str(sys.argv[3])
    save_folder = './dataset/Namiki/drug/'
    datamodule = Large_PretrainingDataset(save_folder,use_conf=True)

    model = PreModel_Container(119,128,n_layers,in_edge_channel=12)

    earlystopping_tracking = 'trn_loss'
    earlystopping_mode = 'min'
    earlystopping_min_delta = 0.0001

    save_model_folder = f'./model_checkpoints/molconf_epoch_{n_epochs}_layers_{n_layers}_{split_strat}/'

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
                    ,enable_progress_bar=True
                    )
    trainer.fit(model, datamodule=datamodule,)

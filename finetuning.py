import os
import warnings
import numpy as np
import yaml
import pandas as pd
import yaml
warnings.filterwarnings('ignore')
import argparse
import torch as t 
from torch import Tensor
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import Trainer
from PreModel import PreModel_Container, SMILEMAE
from LinEvalModel import DeepDrug_Container
from dataset import DeepDrug_Dataset
from utils import * 

def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-cf','--configfile',type=str,default='')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    configfile = args.configfile
    with open(configfile, 'r') as f:
        config = yaml.safe_load(f)
        print(config)
    args = Struct(**config)
    

    entry1_data_folder = '/'.join(args.entry1_file.split('/')[:-2])
    entry2_data_folder = '/'.join(args.entry2_file.split('/')[:-2])
    entry2_seq_file = args.entry2_seq_file
    entry1_seq_file = args.entry1_seq_file
    assert os.path.exists(entry1_seq_file),'file does not exist: %s.'%entry1_seq_file 
    assert os.path.exists(entry2_seq_file),'file does not exist: %s.'%entry2_seq_file
    entry_pairs_file = args.pair_file
    pair_labels_file = args.label_file 
    save_folder = args.save_folder 
    dataset = args.dataset
    save_model_folder = pathjoin(save_folder,'models')
    y_true_file = pathjoin(save_folder,'test_true.csv')
    y_pred_file = pathjoin(save_folder,'test_pred.csv')
    os.makedirs(save_folder,exist_ok=True)
    os.makedirs(pathjoin(save_folder,'plots'),exist_ok=True)
    task_type = args.task
    dataset = args.dataset 
    gpus = args.gpus 
    category  = args.category 
    num_out_dim = args.num_out_dim
    split_strat = args.split_strategy
    model_type = args.model_type
    gconv_ckpt = args.__dict__.get('gconv_ckpt')
    cnn_ckpt = args.__dict__.get('cnn_ckpt')
    lin_Eval = args.lin_eval
    n_layers = args.n_layers

    y_transfrom_func = None

    if args.task in ['binary','multiclass','multilabel']:
        scheduler_ReduceLROnPlateau_tracking = 'F1'
        earlystopping_tracking = "val_epoch_F1"
    else:
        earlystopping_tracking='val_loss'
        scheduler_ReduceLROnPlateau_tracking= 'mse'

    kwargs_dict = dict(save_folder=save_folder,task_type=task_type,
            gpus=gpus,
            entry1_data_folder=entry1_data_folder,
            entry2_data_folder=entry2_data_folder,entry_pairs_file=entry_pairs_file,
            pair_labels_file=pair_labels_file,
            entry1_seq_file =entry1_seq_file ,entry2_seq_file = entry2_seq_file,
            y_true_file=y_true_file,y_pred_file=y_pred_file,
            y_transfrom_func=y_transfrom_func,
            earlystopping_tracking=earlystopping_tracking,
            scheduler_ReduceLROnPlateau_tracking=scheduler_ReduceLROnPlateau_tracking,
            split_strat = split_strat,
            model_type = model_type,
            )
            


    ######  for original training ##########
    _ = print_args(**kwargs_dict)

    datamodule = DeepDrug_Dataset(entry1_data_folder,entry2_data_folder,entry_pairs_file,
                        pair_labels_file,
                        task_type = task_type,
                        y_transfrom_func=y_transfrom_func,
                        entry2_seq_file = entry2_seq_file,
                        entry1_seq_file = entry1_seq_file,
                        category=category,
                        split_strat=split_strat,
                        )

    model =  DeepDrug_Container(
                            task_type = task_type,category=category,
                            scheduler_ReduceLROnPlateau_tracking=scheduler_ReduceLROnPlateau_tracking,
                            num_out_dim = num_out_dim, model_type=model_type,
                            n_layers = n_layers
                            )

    if gconv_ckpt and cnn_ckpt:
        empty_rgcn = PreModel_Container(119,128,n_layers,4,2,encoder_type='deepgcn',decoder_type='deepgcn',loss_fn='mse')
        empty_rgcn.load_from_checkpoint(gconv_ckpt)
        model.model.gconv1.load_state_dict(empty_rgcn.model.encoder.state_dict())
        model.model.gconv2.load_state_dict(empty_rgcn.model.encoder.state_dict())

        empty_conv = SMILEMAE(67,128)
        empty_conv.load_from_checkpoint(cnn_ckpt)
        model.model.gconv1_seq.conv.load_state_dict(empty_conv.model.encoder.conv.state_dict())
        model.model.gconv2_seq.conv.load_state_dict(empty_conv.model.encoder.conv.state_dict())

    if lin_Eval:
        for param in model.model.gconv1.parameters():
            param.requires_grad = False
        for param in model.model.gconv2.parameters():
            param.requires_grad = False
        for param in model.model.gconv1_seq.conv.parameters():
            param.requires_grad = False
        for param in model.model.gconv2_seq.conv.parameters():
            param.requires_grad = False

    if earlystopping_tracking in ['val_loss',]:
        earlystopping_tracking = earlystopping_tracking
        earlystopping_mode = 'min'
        earlystopping_min_delta = 0.0001
    elif earlystopping_tracking in ['val_epoch_F1','val_epoch_auPRC']:
        earlystopping_mode = 'max'
        earlystopping_min_delta = 0.001
    else:
        raise 
    checkpoint_callback = pl_callbacks.ModelCheckpoint(dirpath=save_model_folder,
                                        mode = earlystopping_mode,
                                        monitor=earlystopping_tracking,
                                        save_top_k=1,save_last=True,)
    earlystop_callback = pl_callbacks.EarlyStopping(earlystopping_tracking,verbose=True,
                                        mode = earlystopping_mode,
                                        min_delta=earlystopping_min_delta,
                                        patience=10,)
    
    trainer = Trainer(
                    gpus=[gpus,],
                    accelerator=None,
                    max_epochs=50, min_epochs=5,
                    default_root_dir= save_folder,
                    fast_dev_run=False,
                    check_val_every_n_epoch=1,
                    callbacks=  [checkpoint_callback,
                                earlystop_callback,],
                    enable_progress_bar=False,
                    )
    trainer.fit(model, datamodule=datamodule,)


    ################  Prediction ##################
    print('loading best weight in %s ...'%(checkpoint_callback.best_model_path))
    model = model.load_from_checkpoint(checkpoint_callback.best_model_path,verbose=True)
    model.eval()    
    
    trainer.test(model,dataloaders=datamodule.test_dataloader(),)
    y_pred = trainer.predict(model,dataloaders =datamodule.test_dataloader())
    y_true = np.array(datamodule.pair_labels[datamodule.test_indexs])

    if isinstance(y_pred[0],t.Tensor):
        y_pred = [x.cpu().data.numpy() for x in y_pred]
    if isinstance(y_pred,t.Tensor):
        y_pred = y_pred.cpu().data.numpy()
    y_pred = np.concatenate(y_pred,axis=0) 
    pd.DataFrame(y_pred).to_csv(y_pred_file,header=True,index=False)
    pd.DataFrame(y_true).to_csv(y_true_file,header=True,index=False)
    print('save prediction completed.')

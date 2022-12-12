from self_supervised.datasets import *
from self_supervised.model import SSLM, MetricTracker
from self_supervised.support.visualization import plot_history
from pytorch_lightning.callbacks import EarlyStopping
from tqdm import tqdm
import pytorch_lightning as pl
import os
import numpy as np
import random


def get_trainer(stopping_threshold, epochs, reload_dataloaders_every_n_epochs):
    cb = MetricTracker()
    early_stopping = EarlyStopping(
        monitor="val_loss",
        stopping_threshold=stopping_threshold,
        mode='min',
        patience=5
    )
    trainer = pl.Trainer(
        callbacks= [cb, early_stopping],
        accelerator='auto', 
        devices=1, 
        max_epochs=epochs,
        check_val_every_n_epoch=1,
        reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs)
    return trainer, cb


def training_pipeline(
        dataset_dir:str, 
        root_outputs_dir:str, 
        subject:str,
        imsize:tuple=CONST.DEFAULT_IMSIZE(),
        polygoned=False,
        distortion=False,
        colorized_scar=False,
        patch_localization:bool=False,
        batch_size:int=CONST.DEFAULT_BATCH_SIZE(),
        train_val_split:float=CONST.DEFAULT_TRAIN_VAL_SPLIT(),
        seed:int=CONST.DEFAULT_SEED(),
        projection_training_lr:float=CONST.DEFAULT_LEARNING_RATE(),
        projection_training_epochs:int=CONST.DEFAULT_EPOCHS(),
        fine_tune_lr:float=0.001,
        fine_tune_epochs:int=20):
    
    if patch_localization:
        result_path = root_outputs_dir+subject+'/patch_level/'
    else:
        result_path = root_outputs_dir+subject+'/image_level/'
        
    checkpoint_name = 'best_model.ckpt'
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    print('result dir:', result_path)
    print('checkpoint name:', checkpoint_name)
    print('polygoned:', polygoned)
    print('distortion:', distortion)
    print('colorized scar:', colorized_scar)
    print('patch localization:', patch_localization)
    
    np.random.seed(seed)
    random.seed(seed)
    
    print('>>> preparing datamodule')
    datamodule = GenerativeDatamodule(
        dataset_dir+subject+'/',
        imsize=imsize,
        batch_size=batch_size,
        train_val_split=train_val_split,
        seed=seed,
        duplication=True,
        patch_localization=patch_localization,
        polygoned=polygoned,
        distortion=distortion,
        colorized_scar=colorized_scar
    )
    
    print('>>> setting up the model')
    pretext_model = SSLM(
        num_epochs=projection_training_epochs, 
        lr=projection_training_lr,
        dims=[512,512,512,512,512,512,512,512,512])
    trainer, cb = get_trainer(0.1, projection_training_epochs, 15)
    print('>>> start training (training projection head)')
    trainer.fit(pretext_model, datamodule=datamodule)
    print('>>> training plot')
    plot_history(cb.log_metrics, result_path, mode='training')
    
    print('>>> setting up the model (fine tune whole net)')
    pretext_model.lr = fine_tune_lr
    pretext_model.num_epochs = fine_tune_epochs
    pretext_model.unfreeze_layers(True)
    trainer, cb = get_trainer(0.1, fine_tune_epochs, 20)
    print('>>> start training (fine tune whole net)') 
    trainer.fit(pretext_model, datamodule=datamodule)
    trainer.save_checkpoint(result_path+checkpoint_name)
    print('>>> training plot')
    plot_history(cb.log_metrics, result_path, mode='fine_tune')


def run(
        experiments_list:list,
        dataset_dir:str, 
        root_outputs_dir:str,
        imsize:tuple=CONST.DEFAULT_IMSIZE(),
        polygoned=True,
        distortion=False,
        colorized_scar=False,
        patch_localization:bool=False,
        batch_size:int=CONST.DEFAULT_BATCH_SIZE(),
        train_val_split:float=CONST.DEFAULT_TRAIN_VAL_SPLIT(),
        seed:int=CONST.DEFAULT_SEED(),
        projection_training_lr:float=CONST.DEFAULT_LEARNING_RATE(),
        projection_training_epochs:int=CONST.DEFAULT_EPOCHS(),
        fine_tune_lr:float=0.001,
        fine_tune_epochs:int=20):
    
    os.system('clear')
    pbar = tqdm(range(len(experiments_list)))
    
    for i in pbar:
        pbar.set_description('Training pipeline | current subject is '+experiments_list[i].upper())
        subject = experiments_list[i]
        training_pipeline(
            dataset_dir=dataset_dir, 
            root_outputs_dir=root_outputs_dir, 
            subject=subject,
            imsize=imsize,
            polygoned=polygoned,
            distortion=distortion,
            colorized_scar=colorized_scar,
            patch_localization=patch_localization,
            batch_size=batch_size,
            train_val_split=train_val_split,
            seed=seed,
            projection_training_lr=projection_training_lr,
            projection_training_epochs=projection_training_epochs,
            fine_tune_lr=fine_tune_lr,
            fine_tune_epochs=fine_tune_epochs
            )
        os.system('clear')

if __name__ == "__main__":

    experiments = get_all_subject_experiments('dataset/')
    run(
        experiments_list=['toothbrush', 'transistor', 'zipper'],
        dataset_dir='dataset/', 
        root_outputs_dir='brutta_copia/computations/',
        imsize=(256,256),
        polygoned=True,
        distortion=False,
        colorized_scar=False,
        patch_localization=False,
        batch_size=96,
        train_val_split=0.2,
        seed=0,
        projection_training_lr=0.003,
        projection_training_epochs=30,
        fine_tune_lr=0.001,
        fine_tune_epochs=20
    )
        
        
        
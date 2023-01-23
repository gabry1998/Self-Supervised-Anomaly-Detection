import shutil
from self_supervised.datasets import PretextTaskDatamodule
from self_supervised.models import PeraNet
from self_supervised.custom_callbacks import MetricTracker
from self_supervised.functional import get_all_subject_experiments
from self_supervised.visualization import plot_history
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
import pytorch_lightning as pl
import os
import numpy as np
import random
import torch


def get_trainer(stopping_threshold:float, epochs:int, min_epochs:int, log_dir:str):
    cb = MetricTracker()
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        stopping_threshold=stopping_threshold,
        mode='max',
        patience=7
    )
    mc = ModelCheckpoint(
        dirpath=log_dir, 
        filename='best_model_so_far',
        save_top_k=1, 
        monitor="val_accuracy", 
        mode='max',
        every_n_epochs=5)
    trainer = pl.Trainer(
        default_root_dir=log_dir,
        callbacks= [cb, mc, early_stopping],
        precision=16,
        benchmark=True,
        accelerator='auto', 
        devices=1, 
        max_epochs=epochs,
        min_epochs=min_epochs,
        check_val_every_n_epoch=1)
    return trainer, cb


def training_pipeline(
        dataset_dir:str, 
        root_outputs_dir:str, 
        subject:str,
        imsize:tuple=(256,256),
        polygoned=False,
        colorized_scar=False,
        patch_localization:bool=False,
        batch_size:int=32,
        train_val_split:float=0.2,
        seed:int=0,
        projection_training_lr:float=0.03,
        projection_training_epochs:int=30,
        fine_tune_lr:float=0.001,
        fine_tune_epochs:int=20):
    
    if patch_localization:
        result_path = root_outputs_dir+subject+'/patch_level/'
    else:
        result_path = root_outputs_dir+subject+'/image_level/'
        
    checkpoint_name = 'best_model.ckpt'
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    if os.path.exists(result_path+'logs/'):
        shutil.rmtree(result_path+'logs/')
    
    print('result dir:', result_path)
    print('checkpoint name:', checkpoint_name)
    print('polygoned:', polygoned)
    print('colorized scar:', colorized_scar)
    print('patch localization:', patch_localization)
    
    np.random.seed(seed)
    random.seed(seed)
    
    print('>>> preparing datamodule')
    datamodule = PretextTaskDatamodule(
        subject,
        dataset_dir+subject+'/',
        imsize=imsize,
        batch_size=batch_size,
        train_val_split=train_val_split,
        seed=seed,
        duplication=True,
        patch_localization=patch_localization,
        polygoned=polygoned,
        colorized_scar=colorized_scar,
        patch_size=64
    )
    datamodule.setup()
    pretext_model = PeraNet(
        latent_space_layers=3,
        #latent_space_layers=9,
        num_classes=3, lr=projection_training_lr, num_epochs=projection_training_epochs)
    pretext_model.freeze_net(['backbone'])
    trainer, cb = get_trainer(0.8, projection_training_epochs, min_epochs=3, log_dir=result_path+'logs/')
    print('>>> start training (LATENT SPACE)')
    trainer.fit(pretext_model, datamodule=datamodule)
    print('>>> training plot')
    plot_history(cb.log_metrics, result_path, mode='training')
    #trainer.save_checkpoint(result_path+checkpoint_name)
    print(pretext_model.memory_bank.shape)
    print('>>> setting up the model (fine tune whole net)')
    pretext_model.clear_memory_bank()
    pretext_model.num_epochs = fine_tune_epochs
    pretext_model.lr = fine_tune_lr
    pretext_model.unfreeze()
    trainer, cb = get_trainer(0.999, fine_tune_epochs, min_epochs=15, log_dir=result_path+'logs/')
    print('>>> start training (WHOLE NET)') 
    trainer.fit(pretext_model, datamodule=datamodule)
    trainer.save_checkpoint(result_path+checkpoint_name)
    print(pretext_model.memory_bank.shape)
    print('>>> training plot')
    plot_history(cb.log_metrics, result_path, mode='fine_tune')
    trainer.save_checkpoint(result_path+checkpoint_name)


def run(
        experiments_list:list,
        dataset_dir:str, 
        root_outputs_dir:str,
        imsize:tuple=(256,256),
        polygoned=True,
        colorized_scar=False,
        patch_localization:bool=False,
        batch_size:int=32,
        train_val_split:float=0.2,
        seed:int=0,
        projection_training_lr:float=0.03,
        projection_training_epochs:int=30,
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
        #os.system('clear')


def get_textures_names():
    return ['carpet','grid','leather','tile','wood']

def get_obj_names():
    return np.array([
        'bottle',
        'cable',
        'capsule',
        'hazelnut',
        'metal_nut',
        'pill',
        'screw',
        'tile',
        'toothbrush',
        'transistor',
        'zipper'
    ])

def obj_set_one():
    return [
        'bottle',
        'cable',
        'capsule',
        'hazelnut',
        'metal_nut']

def obj_set_two():
    return [
        'pill',
        'screw',
        'toothbrush',
        'transistor',
        'zipper']

def specials():
    return [
        'cable',
        'capsule',
        'pill',
        'screw']

if __name__ == "__main__":

    experiments = get_all_subject_experiments('dataset/')
    textures = get_textures_names()
    obj1 = obj_set_one()
    obj2 = obj_set_two()
    run(
        experiments_list=['metal_nut'],
        dataset_dir='dataset/', 
        root_outputs_dir='brutta_brutta_copia/computations/',
        imsize=(256,256),
        polygoned=True,
        colorized_scar=True,
        patch_localization=True,
        batch_size=32,
        train_val_split=0.2,
        seed=0,
        projection_training_lr=0.03,
        projection_training_epochs=30,
        fine_tune_lr=0.005,
        fine_tune_epochs=100
    )
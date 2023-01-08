import shutil
from self_supervised.datasets import GenerativeDatamodule
from self_supervised.support import constants as CONST
from self_supervised.model import PeraNet, MetricTracker
from self_supervised.support.functional import get_all_subject_experiments
from self_supervised.support.visualization import plot_history
from pytorch_lightning.callbacks import EarlyStopping
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
        patience=4
    )
    trainer = pl.Trainer(
        default_root_dir=log_dir,
        callbacks= [cb, early_stopping],
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
        imsize:tuple=CONST.DEFAULT_IMSIZE(),
        polygoned=False,
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
    datamodule = GenerativeDatamodule(
        subject,
        dataset_dir+subject+'/',
        imsize=imsize,
        batch_size=batch_size,
        train_val_split=train_val_split,
        seed=seed,
        duplication=True,
        min_dataset_length=500,
        patch_localization=patch_localization,
        polygoned=polygoned,
        colorized_scar=colorized_scar
    )
    
    print('>>> setting up the model')
    pretext_model = PeraNet(
        latent_space_dims=[512,256,128,256,512],
        num_classes=3, lr=projection_training_lr, num_epochs=projection_training_epochs)
    pretext_model.freeze_net(['backbone'])
    trainer, cb = get_trainer(0.95, projection_training_epochs, min_epochs=5, log_dir=result_path+'logs/')
    print('>>> start training (LATENT SPACE)')
    trainer.fit(pretext_model, datamodule=datamodule)
    print('>>> training plot')
    plot_history(cb.log_metrics, result_path, mode='training')
    #trainer.save_checkpoint(result_path+checkpoint_name)
    
    print('>>> setting up the model (fine tune whole net)')
    #pretext_model:PeraNet = PeraNet.load_from_checkpoint(result_path+'best_model.ckpt')
    pretext_model.num_epochs = 20
    pretext_model.lr = 0.005
    pretext_model.unfreeze()
    trainer, cb = get_trainer(0.95, fine_tune_epochs, min_epochs=1, log_dir=result_path+'logs/')
    print('>>> start training (WHOLE NET)') 
    trainer.fit(pretext_model, datamodule=datamodule)
    trainer.save_checkpoint(result_path+checkpoint_name)
    print('>>> training plot')
    plot_history(cb.log_metrics, result_path, mode='fine_tune')
    trainer.save_checkpoint(result_path+checkpoint_name)


def run(
        experiments_list:list,
        dataset_dir:str, 
        root_outputs_dir:str,
        imsize:tuple=CONST.DEFAULT_IMSIZE(),
        polygoned=True,
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


def get_textures_names():
    return np.array(['carpet','grid','leather','tile','wood'])

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
        experiments_list=obj1+obj2,
        dataset_dir='dataset/', 
        root_outputs_dir='brutta_copia/computations/',
        imsize=(256,256),
        polygoned=True,
        colorized_scar=True,
        patch_localization=False,
        batch_size=32,
        train_val_split=0.2,
        seed=0,
        projection_training_lr=0.03,
        projection_training_epochs=30,
        fine_tune_lr=0.01,
        fine_tune_epochs=20
    )
        
        
        
import shutil
from self_supervised.datasets import PretextTaskDatamodule
from self_supervised.models import PeraNet
from self_supervised.custom_callbacks import MetricTracker
from self_supervised.functional import get_all_subject_experiments
from self_supervised.visualization import plot_history
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
from torch.backends import cudnn
from torch import autograd
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
        monitor="val_loss", 
        mode='min',
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
        patch_localization:bool=False,
        batch_size:int=32,
        train_val_split:float=0.2,
        seed:int=0,
        projection_training_lr:float=0.03,
        projection_training_epochs:int=30,
        fine_tune_lr:float=0.001,
        fine_tune_epochs:int=20):
    cudnn.benchmark = True
    autograd.anomaly_mode.set_detect_anomaly(False)
    autograd.profiler.profile(False)
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
        min_dataset_length=1000,
        patch_localization=patch_localization,
        patch_size=64
    )
    datamodule.setup()
    pretext_model = PeraNet()
    pretext_model.compile(
        learning_rate=projection_training_lr,
        epochs=projection_training_epochs
    )
    pretext_model.freeze_net(['backbone'])
    trainer, cb = get_trainer(0.8, projection_training_epochs, min_epochs=3, log_dir=result_path+'logs/')
    print('>>> start training (LATENT SPACE)')
    trainer.fit(pretext_model, datamodule=datamodule)
    print('>>> training plot')
    plot_history(cb.log_metrics, result_path, mode='training')
    pretext_model.clear_memory_bank()
    print('>>> setting up the model (fine tune whole net)')
    pretext_model.num_epochs = fine_tune_epochs
    pretext_model.lr = fine_tune_lr
    pretext_model.unfreeze()
    trainer, cb = get_trainer(0.995, fine_tune_epochs, min_epochs=None, log_dir=result_path+'logs/')
    print('>>> start training (WHOLE NET)') 
    trainer.fit(pretext_model, datamodule=datamodule)
    trainer.save_checkpoint(result_path+checkpoint_name)
    print(pretext_model.memory_bank.shape)
    print('>>> training plot')
    plot_history(cb.log_metrics, result_path, mode='fine_tune')


def run(
        experiments_list:list,
        dataset_dir:str, 
        root_outputs_dir:str,
        imsize:tuple=(256,256),
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
        experiments_list=obj1,
        dataset_dir='dataset/', 
        root_outputs_dir='brutta_brutta_copia/computations/',
        imsize=(256,256),
        patch_localization=True,
        batch_size=64,
        projection_training_lr=0.03,
        projection_training_epochs=30,
        fine_tune_lr=0.01,
        fine_tune_epochs=100
    )
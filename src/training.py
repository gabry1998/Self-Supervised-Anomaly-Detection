from self_supervised.datasets import *
from self_supervised.model import SSLM, MetricTracker
from self_supervised.support.visualization import plot_history
import pytorch_lightning as pl
from tqdm import tqdm
import os


def training_pipeline(
        dataset_dir:str, 
        outputs_dir:str, 
        subject:str,
        level:str,
        patch_type:str='standard',
        imsize=(256,256),
        batch_size=96,
        train_val_split=0.2,
        seed=0,
        lr=0.003,
        epochs=30):
    
    if patch_type=='deformed':
        distortion=True
    else:
        distortion=False
        
    result_path = outputs_dir
    checkpoint_name = 'best_model.ckpt'
    
    if not os.path.exists(result_path):
            os.makedirs(result_path)
    
    print('result dir:', result_path)
    print('checkpoint name:', checkpoint_name)
    print('patch type:', patch_type)
    print('level:', level)
    
    print('>>> preparing datamodule')
    
        
    if level == 'image_level':
        datamodule = GenerativeDatamodule(
            dataset_dir+subject+'/',
            imsize=imsize,
            batch_size=batch_size,
            train_val_split=train_val_split,
            seed=seed,
            duplication=True,
            patch_localization=False,
            distortion=distortion
        )
    else:
        datamodule = GenerativeDatamodule(
            dataset_dir+subject+'/',
            imsize=imsize,
            batch_size=batch_size,
            train_val_split=train_val_split,
            seed=seed,
            duplication=True,
            patch_localization=True,
            distortion=distortion
        )

    
    print('>>> setting up the model')
    pretext_model = SSLM(num_epochs=epochs, lr=lr)
    cb = MetricTracker()
    trainer = pl.Trainer(
        callbacks= [cb],
        accelerator='auto', 
        devices=1, 
        max_epochs=epochs, 
        check_val_every_n_epoch=1,
        reload_dataloaders_every_n_epochs=10)
    print('>>> start training (training projection head)')
    trainer.fit(pretext_model, datamodule=datamodule)
    
    print('>>> training plot')
    plot_history(cb.log_metrics, epochs, result_path, mode='training')
    print('>>> start training (fine tune whole net)')
    pretext_model.lr = 0.001
    pretext_model.num_epochs = 20
    cb = MetricTracker()
    trainer = pl.Trainer(
        callbacks= [cb],
        accelerator='auto', 
        devices=1, 
        max_epochs=20, 
        check_val_every_n_epoch=1,
        reload_dataloaders_every_n_epochs=20)
    pretext_model.unfreeze_layers(True)
    trainer.fit(pretext_model, datamodule=datamodule)
    trainer.save_checkpoint(result_path+checkpoint_name)
    print('>>> training plot')
    plot_history(cb.log_metrics, 20, result_path, mode='fine_tune')

if __name__ == "__main__":

    experiments = [
        ('screw', 'patch_level'),
        ('screw', 'image_level'),
        ('tile', 'patch_level'),
        ('tile', 'image_level'),
        ('toothbrush', 'patch_level'),
        ('toothbrush', 'image_level')
        
    ]
    
    pbar = tqdm(range(len(experiments)))
    for i in pbar:
        pbar.set_description('Pipeline Execution | current subject is '+experiments[i][0].upper())
        training_pipeline(
            dataset_dir='dataset/', 
            outputs_dir='outputs/computations/'+experiments[i][0]+'/'+experiments[i][1]+'/', 
            subject=experiments[i][0],
            level=experiments[i][1],
            imsize=(256,256),
            batch_size=96,
            train_val_split=0.2,
            seed=0,
            lr=0.003,
            epochs=30
            )
        os.system('clear')
        
        
        
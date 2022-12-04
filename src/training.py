from self_supervised.datasets import *
from self_supervised.model import SSLM, MetricTracker
from self_supervised.support.visualization import plot_history
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
from tqdm import tqdm
import os


def get_trainer(stopping_threshold, epochs, reload_dataloaders_every_n_epochs):
    cb = MetricTracker()
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        stopping_threshold=stopping_threshold,
        mode='max',
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
        outputs_dir:str, 
        subject:str,
        patch_localization:bool=False,
        distortion=False,
        imsize=(256,256),
        batch_size=96,
        train_val_split=0.2,
        seed=0,
        lr=0.003,
        epochs=30):
        
    result_path = outputs_dir
    checkpoint_name = 'best_model.ckpt'
    
    if not os.path.exists(result_path):
            os.makedirs(result_path)
    
    print('result dir:', result_path)
    print('checkpoint name:', checkpoint_name)
    print('distortion:', distortion)
    print('patch localization:', patch_localization)
    
    print('>>> preparing datamodule')
    datamodule = GenerativeDatamodule(
        dataset_dir+subject+'/',
        imsize=imsize,
        batch_size=batch_size,
        train_val_split=train_val_split,
        seed=seed,
        duplication=True,
        patch_localization=patch_localization,
        distortion=distortion
    )
    
    print('>>> setting up the model')
    pretext_model = SSLM(num_epochs=epochs, lr=lr)
    trainer, cb = get_trainer(0.90, epochs, 15)
    print('>>> start training (training projection head)')
    trainer.fit(pretext_model, datamodule=datamodule)
    print('>>> training plot')
    plot_history(cb.log_metrics, epochs, result_path, mode='training')
    
    print('>>> setting up the model (fine tune whole net)')
    pretext_model.lr = 0.001
    pretext_model.num_epochs = 20
    pretext_model.unfreeze_layers(True)
    trainer, cb = get_trainer(0.95, 20, 20)
    print('>>> start training (fine tune whole net)') 
    trainer.fit(pretext_model, datamodule=datamodule)
    trainer.save_checkpoint(result_path+checkpoint_name)
    print('>>> training plot')
    plot_history(cb.log_metrics, 20, result_path, mode='fine_tune')

if __name__ == "__main__":

    experiments = get_all_subject_experiments('dataset/', patch_localization=False)
    #experiments = [
    #    ('bottle', False),
    #    ('cable', False),
    #    ('grid', False),
    #    ('toothbrush', False)
    #]
    
    pbar = tqdm(range(len(experiments)))
    for i in pbar:
        pbar.set_description('Pipeline Execution | current subject is '+experiments[i][0].upper())
        
        subject = experiments[i][0]
        patch_localization = experiments[i][1]
        if patch_localization:
            level = 'patch_level'
        else:
            level = 'image_level'
        
        training_pipeline(
            dataset_dir='dataset/', 
            outputs_dir='brutta_copia/computations/'+subject+'/'+level+'/', 
            subject=subject,
            patch_localization=patch_localization,
            distortion=False,
            imsize=(256,256),
            batch_size=96,
            train_val_split=0.2,
            seed=0,
            lr=0.003,
            epochs=30
            )
        os.system('clear')
        
        
        
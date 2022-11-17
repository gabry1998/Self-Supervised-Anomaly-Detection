from self_supervised.datasets import *
from self_supervised.model import SSLM, MetricTracker
from self_supervised.support.visualization import plot_history
import pytorch_lightning as pl
from self_supervised.support.cutpaste_parameters import CPP
import self_supervised.support.constants as CONST
from tqdm import tqdm
import os


def run_pipeline(
        dataset_dir:str, 
        results_dir:str, 
        subject:str, 
        classification_task:str=CONST.DEFAULT_CLASSIFICATION_TASK(), 
        dataset_type_generation:str=CONST.DATASET_GENERATION_TYPES(),
        args:dict=None):

    print('classification task:', classification_task.upper())
    print('dataset type generation:', dataset_type_generation.upper())
    
    result_path = results_dir+subject+'/'+dataset_type_generation+'/'+classification_task+'/'
    checkpoint_name = 'best_model.ckpt'
    
    print('result dir:', result_path)
    print('checkpoint name:', checkpoint_name)
    
    imsize = args['imsize']
    batch_size = args['batch_size']
    train_val_split = args['train_val_split']
    seed = args['seed']
    lr = args['lr']
    epochs = args['epochs']
    
    
    print('>>> preparing datamodule')
    if dataset_type_generation == 'generative_dataset':
        datamodule = GenerativeDatamodule(
            dataset_dir+subject+'/',
            imsize=imsize,
            batch_size=batch_size,
            train_val_split=train_val_split,
            seed=seed,
            classification_task=classification_task,
            duplication=True
        )
    else:
        datamodule = CutPasteClassicDatamodule(
            dataset_dir+subject+'/',
            imsize=imsize,
            batch_size=batch_size,
            train_val_split=train_val_split,
            classification_task=classification_task,
            seed=seed,
            duplication=True
        )
    
    print('>>> setting up the model')
    pretext_model = SSLM(classification_task, lr=lr, seed=seed)
    cb = MetricTracker()
    trainer = pl.Trainer(
        callbacks= [cb],
        accelerator='auto', 
        devices=1, 
        max_epochs=epochs, 
        check_val_every_n_epoch=1,
        reload_dataloaders_every_n_epochs=10)
    print('>>> start training')
    trainer.fit(pretext_model, datamodule=datamodule)
    trainer.save_checkpoint(result_path+checkpoint_name)
    
    print('>>> training plot')
    plot_history(cb.log_metrics, epochs, result_path, classification_task)
    

if __name__ == "__main__":
    dataset_dir = '/home/ubuntu/TesiAnomalyDetection/dataset/'
    results_dir = '/home/ubuntu/TesiAnomalyDetection/outputs/computations/'
    
    imsize= (256,256)
    batch_size = 64
    train_val_split = 0.2
    seed = 0
    lr = 0.001
    epochs = 30
    
    args = {
        'imsize': imsize,
        'batch_size': batch_size,
        'train_val_split': train_val_split,
        'seed': seed,
        'lr': lr,
        'epochs': epochs
    }
    experiments = [
        #('screw', '3-way', 'generative_dataset'),
        #('toothbrush', '3-way', 'generative_dataset')
        ('bottle', '3-way', 'generative_dataset')
    ]
    pbar = tqdm(range(len(experiments)))
    for i in pbar:
        pbar.set_description('Pipeline Execution | current subject is '+experiments[i][0].upper())
        run_pipeline(
            dataset_dir, 
            results_dir, 
            experiments[i][0], 
            experiments[i][1], 
            experiments[i][2],
            args)
        os.system('clear')
        
        
        
from self_supervised.datasets import *
from self_supervised.model import SSLM, MetricTracker
from self_supervised.support.visualization import plot_history
import pytorch_lightning as pl
from self_supervised.support.cutpaste_parameters import CPP


def run_pipeline(
        dataset_dir, 
        results_dir, 
        subject, 
        classification_task='binary', 
        dataset_type_generation='classic_dataset'):
    
    print('#################')
    print('>>> running pipeline ('+subject.upper()+')')
    print('classification task:', classification_task)
    print('dataset type generation:', dataset_type_generation)
    print('cutpaste parameters:')
    print(CPP.summary)
    
    result_path = results_dir+subject+'/'+dataset_type_generation+'/'+classification_task+'/'
    checkpoint_name = subject+'.ckpt'
    
    print('result dir:', result_path)
    print('checkpoint name:', checkpoint_name)
    
    imsize=(256,256)
    batch_size = 64
    train_val_split = 0.2
    seed = 0
    lr = 0.001
    epochs = 30
    
    print('image size:', imsize)
    print('batch size:', batch_size)
    print('split rate:', train_val_split)
    print('seed:', seed)
    print('optimizer:', 'SGD')
    print('learning rate:', lr)
    print('epochs:', epochs)
    
    
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
    print(type(datamodule))
    
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

    trainer.fit(pretext_model, datamodule=datamodule)
    trainer.save_checkpoint(result_path+checkpoint_name)
    
    print('>>> training plot')
    plot_history(cb.log_metrics, epochs, result_path, classification_task)
        
    print('>>> testing')
    pretext_model = SSLM.load_from_checkpoint(result_path+checkpoint_name, model=pretext_model.model)
    datamodule.setup('test')
    trainer.test(pretext_model, dataloaders=datamodule.test_dataloader())
    

if __name__ == "__main__":
    dataset_dir = '/home/ubuntu/TesiAnomalyDetection/dataset/'
    results_dir = '/home/ubuntu/TesiAnomalyDetection/outputs/computations/'
    #subjects = ['bottle', 'grid', 'screw', 'tile', 'toothbrush']
    subjects = ['bottle', 'grid']
    for subject in subjects:
        run_pipeline(dataset_dir, results_dir, subject, '3-way', 'classic_dataset')
        run_pipeline(dataset_dir, results_dir, subject, '3-way', 'generative_dataset')
        run_pipeline(dataset_dir, results_dir, subject, 'binary', 'classic_dataset')
        run_pipeline(dataset_dir, results_dir, subject, 'binary', 'generative_dataset')
        
        
        
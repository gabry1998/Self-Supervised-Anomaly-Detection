from self_supervised.datasets import GenerativeDatamodule
from self_supervised.model import SSLM, MetricTracker
from self_supervised.support.visualization import plot_history
import pytorch_lightning as pl


def run_pipeline(dataset_dir, results_dir, subject, classification_task='binary'):
    print('#################')
    print('>>> running pipeline ('+subject.upper()+')')
    print('task:', classification_task)
    
    result_path = results_dir+subject+'/'
    checkpoint_name = subject+'_'+classification_task+'_model_weights.ckpt'
    
    imsize=(256,256)
    batch_size = 32
    train_val_split = 0.2
    seed = 0
    lr = 0.001
    epochs = 60
    
    
    print('>>> preparing datamodule')
    datamodule = GenerativeDatamodule(
        dataset_dir+subject+'/',
        imsize=imsize,
        batch_size=batch_size,
        train_val_split=train_val_split,
        seed=seed,
        classification_task=classification_task)
    datamodule.setup('fit')
    
    print('>>> setting up the model')
    pretext_model = SSLM(classification_task, lr=lr, seed=seed)
    cb = MetricTracker()
    trainer = pl.Trainer(
        callbacks= [cb],
        accelerator='auto', 
        devices=1, 
        max_epochs=epochs, 
        check_val_every_n_epoch=1,
        reload_dataloaders_every_n_epochs=20)

    trainer.fit(pretext_model, datamodule=datamodule)
    trainer.save_checkpoint(result_path+checkpoint_name)
    
    print('>>> testing')
    pretext_model = SSLM.load_from_checkpoint(result_path+checkpoint_name, model=pretext_model.model)
    datamodule.setup('test')
    trainer.test(pretext_model, dataloaders=datamodule.test_dataloader())
    
    print('>>> training plot')
    plot_history(cb.log_metrics, epochs, result_path, classification_task)
    

if __name__ == "__main__":
    dataset_dir = '/home/ubuntu/TesiAnomalyDetection/dataset/'
    results_dir = '/home/ubuntu/TesiAnomalyDetection/outputs/computations/'
    #subjects = ['bottle', 'grid', 'screw', 'tile', 'toothbrush']
    subjects = ['bottle']
    for subject in subjects:
        run_pipeline(dataset_dir, results_dir, subject, '3-way')
        run_pipeline(dataset_dir, results_dir, subject, 'binary')
        
        
        
from ssl_datasets.generative_dataset import PretextTaskGenerativeDatamodule
from ssl_models.model import *
from support.visualization import plot_history

def run_pipeline(dataset_dir, results_dir, subject, task='binary'):
    print('#################')
    print('>>> running pipeline ('+subject.upper()+')')
    print('task:', task)
    
    result_path = results_dir+subject+'/'
    checkpoint_name = subject+'_'+task+'_model_weights.ckpt'
    
    imsize=(256,256)
    batch_size = 64
    train_val_split = 0.2
    seed = 0
    lr = 0.001
    epochs = 60
    
    
    print('>>> preparing datamodule')
    datamodule = PretextTaskGenerativeDatamodule(
        dataset_dir+subject+'/',
        imsize=imsize,
        batch_size=batch_size,
        train_val_split=train_val_split,
        seed=seed,
        n_repeat=4,
        pretextask=task)
    datamodule.prepare_data()
    
    print('>>> setting up the model')
    pretext_model = SSLM(task, lr=lr, seed=seed)
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
    
    print('>>> testing')
    pretext_model = SSLM.load_from_checkpoint(result_path+checkpoint_name, model=pretext_model.model)
    datamodule.setup('test')
    trainer.test(pretext_model, dataloaders=datamodule.test_dataloader())
    
    print('>>> training plot')
    plot_history(cb.log_metrics, epochs, result_path, task)
    

if __name__ == "__main__":
    dataset_dir = '/home/ubuntu/TesiAnomalyDetection/dataset/'
    results_dir = '/home/ubuntu/TesiAnomalyDetection/outputs/computations/'
    #subjects = ['bottle', 'grid', 'screw', 'tile', 'toothbrush']
    subjects = ['bottle']
    for subject in subjects:
        run_pipeline(dataset_dir, results_dir, subject, '3-way')
    
    for subject in subjects:
        run_pipeline(dataset_dir, results_dir, subject, 'binary')

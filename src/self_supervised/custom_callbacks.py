import pytorch_lightning as pl



class MetricTracker(pl.Callback):
    def __init__(self):
        
        self.log_metrics = {
            'train':{
                'accuracy':[],
                'loss':[]
            },
            'val':{
                'accuracy':[],
                'loss':[]
            }
        }

    def on_train_epoch_end(self, trainer, pl_module):
        elogs = trainer.logged_metrics
        self.log_metrics['train']['accuracy'].append(elogs['train_accuracy'].item())
        self.log_metrics['train']['loss'].append(elogs['train_loss'].item())
        self.log_metrics['val']['accuracy'].append(elogs['val_accuracy'].item())
        self.log_metrics['val']['loss'].append(elogs['val_loss'].item())


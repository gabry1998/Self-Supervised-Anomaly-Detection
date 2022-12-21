from self_supervised.model import GDE, PeraNet, MetricTracker
from self_supervised.datasets import MVTecDatamodule, GenerativeDatamodule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import numpy as np
import random
import pytorch_lightning as pl
from self_supervised.support.functional import *
from self_supervised.support.visualization import *
import self_supervised.metrics as mtr


def do_inference(model, x):
    if torch.cuda.is_available():
        with torch.no_grad():
            output = model(x.to('cuda'))
            y_hat = output['classifier']
            embeddings = output['latent_space']
    else:
        with torch.no_grad():
            output = model(x.to('cpu'))
            y_hat = output['classifier']
            embeddings = output['latent_space']
    y_hat = y_hat.to('cpu')
    embeddings = embeddings.to('cpu')
    y_hat = get_prediction_class(y_hat)
    
    return y_hat, embeddings


def get_trainer(stopping_threshold:float, epochs:int, min_epochs:int=10):
    cb = MetricTracker()
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        stopping_threshold=stopping_threshold,
        mode='max',
        patience=5
    )
    trainer = pl.Trainer(
        callbacks= [cb, early_stopping],
        precision=16,
        benchmark=True,
        accelerator='auto', 
        devices=1, 
        max_epochs=epochs,
        min_epochs=min_epochs,
        check_val_every_n_epoch=1)
    return trainer, cb

def train():
    datamodule = GenerativeDatamodule(
            'dataset/bottle/',
            imsize=(256,256),
            batch_size=96,
            train_val_split=0.2,
            seed=0,
            duplication=True,
            patch_localization=False,
            polygoned=True,
            colorized_scar=True
    )
    trainer, cb = get_trainer(0.95, 30)
    random.seed(0)
    peranet = PeraNet()
    peranet.freeze_net(['backbone'])
    trainer.fit(peranet, datamodule=datamodule)
    plot_history(cb.log_metrics, 'qui/', mode='training')
    trainer.save_checkpoint('qui/best_model.ckpt')
    peranet:PeraNet = PeraNet.load_from_checkpoint('qui/best_model.ckpt')
    peranet.lr = 0.01
    peranet.num_epochs = 20
    peranet.unfreeze_net()
    trainer, cb = get_trainer(0.95, 20, min_epochs=3)
    trainer.fit(peranet, datamodule=datamodule)
    trainer.save_checkpoint('qui/best_model.ckpt')
    plot_history(cb.log_metrics, 'qui/', mode='training')

    mvtec = MVTecDatamodule(
        'dataset/bottle/',
        'bottle',
        imsize=(256,256),
        batch_size=96,
        seed=0
    )
    peranet.enable_mvtec_inference()
    trainer.test(peranet, mvtec)


def test():
    datamodule = GenerativeDatamodule(
            'dataset/bottle/',
            imsize=(256,256),
            batch_size=128,
            train_val_split=0.2,
            seed=204110176,
            duplication=True,
            patch_localization=False,
            polygoned=True,
            colorized_scar=True
    )
    datamodule.setup('test')
    mvtec = MVTecDatamodule(
        'dataset/bottle/',
        'bottle',
        imsize=(256,256),
        batch_size=128,
        seed=204110176
    )
    mvtec.setup()
    trainer, cb = get_trainer(0.95, 30)
    random.seed(204110176)
    peranet = PeraNet()
    peranet.freeze_net()
    print('>>> Inferencing...')
    predictions_artificial = trainer.predict(peranet, datamodule)[0]

    print('>>> Inferencing over real mvtec images...')
    peranet.enable_mvtec_inference()
    predictions_mvtec = trainer.predict(peranet, mvtec)[0]

    print('>>> Embeddings for GDE..')
    predictions_mvtec_gde_train = trainer.predict(peranet, mvtec.train_dataloader())[0]
    print(predictions_mvtec['embedding'])
    embeddings_mvtec = predictions_mvtec['embedding']
    train_embeddings_gde = predictions_mvtec_gde_train['embedding']
    embeddings_artificial = predictions_artificial['embedding']
    gde = GDE()
    gde.fit(train_embeddings_gde)
    mvtec_test_scores = gde.predict(embeddings_mvtec)
    mvtec_test_labels = predictions_mvtec['y_true']

    print('>>> calculating (IMAGE LEVEL) ROC, AUC, F1..')
    mvtec_test_scores = normalize(mvtec_test_scores)
    fpr, tpr, _ = mtr.compute_roc(mvtec_test_labels, mvtec_test_scores)
    auc_score = mtr.compute_auc(fpr, tpr)
    test_y_hat = multiclass2binary(predictions_mvtec['y_hat'])
    f_score = mtr.compute_f1(torch.tensor(mvtec_test_labels), test_y_hat)
    print('>>> plot ROC..')
    plot_curve(
        fpr, tpr, 
        auc_score, 
        saving_path='qui/',
        title='Roc curve for '+'bottle'.upper()+' ['+str(204110176)+']',
        name='roc.png')
    
    total_y = torch.cat([predictions_artificial['y_tsne'], predictions_mvtec['y_tsne']])
    total_embeddings = torch.cat([predictions_artificial['embedding'], predictions_mvtec['embedding']])
    plot_tsne(
        total_embeddings, 
        total_y, 
        saving_path='qui/', 
        title='Embeddings projection for '+'bottle'.upper()+' ['+str(204110176)+']')

train()   
test()
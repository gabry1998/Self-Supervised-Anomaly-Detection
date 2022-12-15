from self_supervised.gradcam import GradCam
from self_supervised.model import GDE, SSLM
from self_supervised.datasets import *
from tqdm import tqdm
import self_supervised.datasets as dt
import self_supervised.support.constants as CONST
import self_supervised.support.visualization as vis
import self_supervised.metrics as mtr
import time
import random
import numpy as np

dataset_dir='dataset/'
root_inputs_dir='outputs/computations/'
root_outputs_dir='brutta_copia/computations/'
subject='transistor'
np.random.seed(204110176)
random.seed(204110176)
model_dir = root_inputs_dir+subject+'/image_level/'+'best_model.ckpt'
outputs_dir = root_outputs_dir+subject+'/image_level/'

print('')
print('>>> Loading model')
sslm = SSLM(dims=[512,512,512,512,512,512,512,512,512])
sslm:SSLM = SSLM.load_from_checkpoint(model_dir, model=sslm.model)
sslm.eval()
print('>>> Generating test dataset (artificial)')
artificial = GenerativeDatamodule(
    dataset_dir+subject+'/',
    imsize=(256,256),
    batch_size=128,
    seed=204110176,
    duplication=True,
    min_dataset_length=500,
    patch_localization=False,
    polygoned=True,
)
artificial.setup('test')

print('>>> loading mvtec dataset')
mvtec = MVTecDatamodule(
            dataset_dir+subject+'/',
            subject=subject,
            imsize=(256,256),
            batch_size=128,
            seed=0
)
mvtec.setup()

tester = pl.Trainer(
    accelerator='auto', 
    enable_checkpointing=False,
    devices=1)

print('>>> Predicting artificial...')
start = time.time()
predictions_artificial = tester.predict(sslm, artificial.test_dataloader())[0]
end = time.time() - start
print('Done in '+str(end)+ 'sec')

print('>>> Predicting mvtec...')
sslm.mvtec = True
start = time.time()
predictions_mvtec = tester.predict(sslm, mvtec.test_dataloader())[0]
end = time.time() - start
print('Done in '+str(end)+ 'sec')


print('>>> Train Embeddings for GDE..')
start = time.time()
gde_train_mvtec = tester.predict(sslm, mvtec.train_dataloader())[0]
end = time.time() - start
print('Done in '+str(end)+ 'sec')

test_mvtec = torch.nn.functional.normalize(predictions_mvtec['embedding'], p=2, dim=1)
train_mvtec = torch.nn.functional.normalize(gde_train_mvtec['embedding'], p=2, dim=1)
artificial_embeds = torch.nn.functional.normalize(predictions_artificial['embedding'], p=2, dim=1)

print('>>> Train Embeddings for GDE..')
start = time.time()
gde = GDE()
gde.fit(train_mvtec)
mvtec_test_scores = gde.predict(test_mvtec)
end = time.time() - start
print('Done in '+str(end)+ 'sec')

print('>>> calculating ROC, AUC, F1..')
start = time.time()
mvtec_test_scores = normalize(mvtec_test_scores)
fpr, tpr, _ = mtr.compute_roc(predictions_mvtec['y_true'], mvtec_test_scores)
auc_score = mtr.compute_auc(fpr, tpr)
f_score = mtr.compute_f1(predictions_mvtec['y_true'], predictions_mvtec['y_hat'])
end = time.time() - start
print('Done in '+str(end)+ 'sec')

print('>>> plot ROC..')
vis.plot_curve(
    fpr, tpr, 
    auc_score, 
    saving_path=outputs_dir,
    title='Roc curve for '+subject.upper()+' ['+str(204110176)+']',
    name='roc.png')

print('>>> Generating tsne visualization')
total_y = torch.cat([predictions_artificial['y_true'], predictions_mvtec['y_true_tsne']])
total_embeddings = torch.cat([artificial_embeds, test_mvtec])
vis.plot_tsne(
    total_embeddings, 
    total_y, 
    saving_path=outputs_dir, 
    title='Embeddings projection for '+subject.upper()+' ['+str(204110176)+']')

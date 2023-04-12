from self_supervised.datasets import PretextTaskDatamodule, MVTecDatamodule
from self_supervised.models import PeraNet
from self_supervised.custom_callbacks import MetricTracker
from torch import Tensor
from torch.nn.functional import softmax
from self_supervised.constants import ModelOutputsContainer, EvaluationOutputContainer
from self_supervised.converters import imagetensor2array, multiclass2binary
from self_supervised.functional import get_prediction_class, get_all_subject_experiments
from self_supervised import visualization as vis
from self_supervised import metrics as mtr
from PIL import Image, ImageDraw
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.backends import cudnn
from torch import autograd

import os
import shutil
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import pytorch_lightning as pl



class Evaluator:
    def __init__(
            self, 
            evaluation_metrics:list=['image_auroc','pixel_auroc','aupro'], 
            visualize_tsne:bool=True,
            tsne_labels:list=['good','big_polygon','small_rectangles','segment']) -> None:
        
        self.evaluation_metrics = evaluation_metrics
        self.tsne = visualize_tsne
        self.tsne_labels = tsne_labels
        self.scores = EvaluationOutputContainer() 
        
    def evaluate(
            self, 
            outputs_artificial:ModelOutputsContainer, 
            outputs_real:ModelOutputsContainer, 
            subject:str, 
            outputs_dir:str, 
            seed:int=None):
        # tsne visualization
        if self.tsne:
            vis.plot_tsne(
                torch.cat([outputs_artificial.embedding_vectors, outputs_real.embedding_vectors]),
                torch.cat([outputs_artificial.y_tsne, outputs_real.y_tsne]),
                saving_path=outputs_dir,
                title=subject.upper()+' feature visualization',
                name=subject+'_tsne.png')
        
        # computing image auroc
        if 'image_auroc' in self.evaluation_metrics:
            # y_true and y_hat are both a flat vector
            fpr, tpr, _ = mtr.compute_roc(outputs_real.y_true, outputs_real.y_hat)
            image_auc_score = mtr.compute_auc(fpr, tpr)
            vis.plot_curve(
                fpr, tpr, 
                image_auc_score, 
                saving_path=outputs_dir,
                title='Roc curve for '+subject.upper()+' ['+str(seed)+']',
                name=subject+'_image_roc.png')
        
        # computing pixel auroc
        fpr, tpr, _ = mtr.compute_roc(outputs_real.y_true, outputs_real.y_hat)
        pixel_auc_score = mtr.compute_auc(fpr, tpr)
        vis.plot_curve(
            fpr, tpr, 
            pixel_auc_score, 
            saving_path=outputs_dir,
            title='Roc curve for '+subject.upper()+' ['+str(seed)+']',
            name=subject+'_pixel_roc.png')
        
        # computing aupro
        all_fprs, all_pros = mtr.compute_pro(
            anomaly_maps=np.array(outputs_real.anomaly_maps),
            ground_truth_maps=np.array(outputs_real.ground_truths))
        aupro = mtr.compute_aupro(all_fprs, all_pros, 0.3)
        vis.plot_curve(
            all_fprs,
            all_pros,
            aupro,
            saving_path=outputs_dir,
            title='Pro curve for '+subject.upper()+' ['+str(seed)+']',
            name=subject+'_pro.png'
        )
                  
        
class ErrorAnalyzer:
    def __init__(
            self,
            raw_predictions:Tensor,
            true_labels:Tensor,
            tensor_images:Tensor,
            pretty_labels:list=None) -> None:
        
        self.probabilities = softmax(raw_predictions, dim=1)
        self.true_labels = true_labels
        self.tensor_images = tensor_images
        self.pretty_labels = pretty_labels
        
    def analyze(
            self, 
            num_images_to_analyze:int=10, 
            randomized:bool=True, 
            output_path:str='probabilities.png'):
        y_hat = multiclass2binary(get_prediction_class(self.probabilities))
        probabilities = np.asarray(self.probabilities)
        wrong_idx = (y_hat != self.true_labels).nonzero()[:, 0]
        
        if randomized:
            sample = random.choices(wrong_idx, k=num_images_to_analyze)
        else:
            sample = wrong_idx[0:num_images_to_analyze-1]
        
        fig, axs = plt.subplots(1,num_images_to_analyze, figsize=(64,64))
        
        for i in range(num_images_to_analyze):
            idx = sample[i]
            probs = probabilities[idx]
            image_tensor = self.tensor_images[idx]
            image_array = Image.fromarray(imagetensor2array(image_tensor)).convert('RGB')
            
            # text annotations
            prediction_notes= ''
            for j in range(len(self.pretty_labels)):
                prediction_notes += self.pretty_labels[j]+': '+str(probs[j])+'\n'
            true_label = 'GOOD' if self.true_labels[idx]==0 else 'DEFECT'
            predicted_label = 'GOOD' if y_hat[idx]==0 else 'DEFECT'
            prediction_notes += '\nTrue label: '+true_label+'\n'
            prediction_notes += 'Predicted label: '+predicted_label
            textbox = Image.new('RGB', (image_array.size[0], int(image_array.size[0]/2)), color='white')
            draw  = ImageDraw.Draw(textbox)
            draw.text((0,0), prediction_notes, fill='black')
            
            tot = np.vstack([np.array(textbox), np.array(image_array)])
            axs[i].imshow(tot)
            axs[i].axis('off')
        
        plt.savefig(output_path, bbox_inches='tight')
        

def training(
        dataset_dir:str,
        outputs_dir:str,
        subject:str,
        imsize:tuple=(256,256),
        patch_localization:bool=False,
        patchsize:int=32,
        seed:int=0,
        batch_size:int=96,
        projection_training:tuple=(10,0.03),
        fine_tune:tuple=(30,0.005)
        ):
    
    
    autograd.anomaly_mode.set_detect_anomaly(False)
    autograd.profiler.profile(False)
    
    checkpoint_name = 'best_model.ckpt'
    proj_epochs, proj_lr = projection_training
    fine_tune_epochs, fine_tune_lr = fine_tune
    
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    if os.path.exists(outputs_dir+'logs/'):
        shutil.rmtree(outputs_dir+'logs/')
        
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print('>>> preparing datamodule')
    datamodule = PretextTaskDatamodule(
        subject,
        dataset_dir+subject+'/',
        imsize=imsize,
        batch_size=batch_size,
        seed=seed,
        patch_localization=patch_localization,
        patch_size=patchsize
    )
    datamodule.setup()
    
    print('>>> preparing model')
    pretext_model = PeraNet(
        learning_rate=proj_lr, 
        epochs=proj_epochs)
    pretext_model.freeze_net(['backbone'])
    
    cb = MetricTracker()
    
    trainer = pl.Trainer(
        default_root_dir=outputs_dir+'logs/',
        callbacks= [cb],
        precision=16,
        benchmark=True,
        accelerator='auto', 
        devices=1, 
        max_epochs=proj_epochs,
        check_val_every_n_epoch=1)
    
    trainer.fit(pretext_model, datamodule=datamodule)
    print('>>> training plot')
    vis.plot_history(cb.log_metrics, outputs_dir, mode='training')
    pretext_model.clear_memory_bank()
    trainer.save_checkpoint(outputs_dir+checkpoint_name, weights_only=True)
    
    print('>>> setting up the model (fine tune whole net)')
    pretext_model:PeraNet = PeraNet().load_from_checkpoint(
        outputs_dir+checkpoint_name, 
        learning_rate=fine_tune_lr,
        epochs=fine_tune_epochs,
        stage='fine_tune')
    pretext_model.unfreeze()
    print(pretext_model.lr, pretext_model.num_epochs)
    
    mc = ModelCheckpoint(
        dirpath=outputs_dir+'logs/', 
        filename='best_model_so_far',
        save_top_k=1, 
        monitor="val_loss", 
        mode='min',
        every_n_epochs=5)
    cb = MetricTracker()
    trainer = pl.Trainer(
        default_root_dir=outputs_dir+'logs/',
        callbacks= [cb, mc],
        precision=16,
        benchmark=True,
        accelerator='auto', 
        devices=1, 
        max_epochs=fine_tune_epochs,
        check_val_every_n_epoch=1)
    print('>>> start training (WHOLE NET)') 
    trainer.fit(pretext_model, datamodule=datamodule)
    trainer.save_checkpoint(outputs_dir+checkpoint_name)
    print(pretext_model.memory_bank.shape)
    print('>>> training plot')
    vis.plot_history(cb.log_metrics, outputs_dir, mode='fine_tune')
import shutil
from self_supervised.gradcam import GradCam
from self_supervised.model import GDE, PeraNet
from self_supervised.datasets import MVTecDatamodule, GenerativeDatamodule
from tqdm import tqdm
from torch import Tensor
import self_supervised.support.constants as CONST
from self_supervised.support.functional import \
    get_all_subject_experiments, get_prediction_class, heatmap2mask, multiclass2binary, normalize
import self_supervised.support.visualization as vis
import self_supervised.metrics as mtr
import time
import random
import numpy as np
import torch
import pytorch_lightning as pl
import os



class Evaluator:
    def __init__(
            self,
            dataset_dir:str=None,
            root_input_dir:str=None, 
            root_output_dir:str=None,
            subject:str=None,
            model_name:str='best_model.ckpt',
            patch_localization=True,
            patch_dim:int=32,
            stride:int=4,
            seed=0,
            threshold:int=0.5) -> None:
        
        mode = 'patch_level' if patch_localization else 'image_level'
        self.dataset_dir = dataset_dir
        self.model_dir = root_input_dir+subject+'/'+mode+'/'
        self.subject = subject
        self.root_input_dir = root_input_dir,
        self.root_output_dir = root_output_dir
        self.outputs_dir = root_output_dir+subject+'/'+mode+'/gradcam/'
        
        self.model_name = model_name
        self.patch_localization = patch_localization
        self.patch_dim = patch_dim
        self.stride = stride
        self.threshold = threshold
        
        random.seed(seed)
        np.random.seed(seed)
        
        
    def setup_dataset(self, imsize:tuple=(256,256), batch_size:int=128):
        
        self.imsize = imsize
        
        self.artificial_datamodule = GenerativeDatamodule(
        self.subject,
        self.dataset_dir+self.subject+'/',
        imsize=imsize,
        batch_size=batch_size,
        seed=self.seed,
        duplication=True,
        min_dataset_length=500,
        patch_localization=False
        )
        self.artificial_datamodule.setup('test')
        
        
        self.mvtec_datamodule = MVTecDatamodule(
            root_dir=self.dataset_dir+self.subject+'/',
            subject=self.subject,
            imsize=imsize,
            batch_size=64
        )
        self.mvtec_datamodule.setup()
    
    
    def evaluate(self):
        peranet:PeraNet = PeraNet.load_from_checkpoint(
            self.model_dir+self.model_name
        )
        if not self.patch_localization:
            tester = pl.Trainer(accelerator='auto', devices=1)
            # artificial dataset inference
            artificial_predictions = tester.predict(peranet, self.artificial_datamodule)[0]
            artificial_y_hat = artificial_predictions['y_hat']
            artificial_y_true = artificial_predictions['y_true']
            artificial_embeddings = artificial_predictions['embeddings']
            artificial_tsne = artificial_predictions['y_hat_tsne']
            
            # mvtec inference
            peranet.enable_mvtec_inference()
            mvtec_predictions = tester.predict(peranet, self.mvtec_datamodule)[0]
            mvtec_x = mvtec_predictions['x_prime']
            mvtec_y_hat = mvtec_predictions['y_hat']
            mvtec_y_true = mvtec_predictions['y_true']
            mvtec_groundtruths = mvtec_predictions['groundtruth']
            mvtec_embeddings = mvtec_predictions['embeddings']
            mvtec_tsne = mvtec_predictions['y_hat_tsne']
            
            # train data for kde
            kde_train_data = tester.predict(peranet, self.mvtec_datamodule.train_dataloader())[0]
            
            # normalization
            kde_train_data = torch.nn.functional.normalize(kde_train_data, p=2, dim=1)
            artificial_embeddings = torch.nn.functional.normalize(artificial_embeddings, p=2, dim=1)
            mvtec_embeddings = torch.nn.functional.normalize(mvtec_embeddings, p=2, dim=1)
            
            # computing anomaly scores
            kde = GDE()
            kde.fit(kde_train_data)
            anomaly_scores = kde.predict(mvtec_embeddings)
            
            # gradcam for pixel metrics and pro
            gradcam = GradCam(model=peranet)
            anomaly_maps = []
            for i in range(len(len(artificial_tsne))):
                if mvtec_y_hat[i] == 0:
                    saliency_map = torch.zeros(self.imsize)[None, :]
                else:
                    x = mvtec_x[i]
                    # saliency is a 1x1xHxW tensor here
                    saliency_map = gradcam(x[None, :], mvtec_y_hat[i])

                anomaly_maps.append(np.array(saliency_map.squeeze())) # saliency is a 1xHxW tensor here
            groundtruth_maps = mvtec_groundtruths.squeeze()
        
        # computing metrics
        anomaly_scores = normalize(anomaly_scores)
        fpr, tpr, auc_score = self._compute_auroc(mvtec_y_true, anomaly_scores)
        f1 = self._compute_f1(mvtec_y_true, mvtec_y_hat)
        all_fprs, all_pros, au_pro = self._compute_aupro(groundtruth_maps, anomaly_maps)
        
        if not self.patch_localization:
            self._compute_tsne()
        

    def _compute_aupro(self, targets:Tensor, scores:Tensor):
        # target and scores are both a 2d map
        all_fprs, all_pros = mtr.compute_pro(
            anomaly_maps=np.array(scores),
            ground_truth_maps=np.array(targets))

        au_pro = mtr.compute_aupro(all_fprs, all_pros, 0.3)
        return all_fprs, all_pros, au_pro
    
    
    def _compute_f1(self, targets:Tensor, scores:Tensor):
        # target and scores are both a flat vector
        return mtr.compute_f1(targets, scores)
    
        
    def _compute_auroc(self, targets:Tensor, scores:Tensor):
        # target and scores are both a flat vector
        fpr, tpr, _ = mtr.compute_roc(targets, scores)
        auc_score = mtr.compute_auc(fpr, tpr)
        return fpr, tpr, auc_score
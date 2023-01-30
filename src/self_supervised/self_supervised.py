from tqdm import tqdm
from abc import ABC, abstractmethod
from custom_callbacks import MetricTracker
from models import AnomalyDetector, PeraNet
from datasets import PretextTaskDatamodule, MVTecDatamodule
from self_supervised.functional import extract_patches, get_prediction_class
from visualization import plot_history
from torchvision.transforms import functional
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import Tensor

import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import numpy as np
import random
import os
import metrics as mtr
import visualization as vis



class __BaseClassWrapper(ABC):
    def __init__(
            self,
            subject:str,
            dataset_directory:str,
            model_input_directory:str=None,
            output_directory:str='./',
            seed:int=0) -> None:
        
        assert(subject is not None)
        assert(dataset_directory is not None)
        assert(seed is not None)
        if output_directory and not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        self.subject = subject
        self.dataset_directory = dataset_directory
        self.model_input_directory = model_input_directory
        self.output_directory = output_directory
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
    
    
    @abstractmethod
    def setup_dataset(self):
        pass
    
    
    @abstractmethod
    def setup_model(self):
        pass
    
    
    @abstractmethod
    def execute(self):
        pass
    
    
class Trainer(__BaseClassWrapper):
    def __init__(
            self, 
            subject:str,
            dataset_directory: str, 
            model_input_directory: str = None, 
            output_directory: str = './', 
            seed: int = 0,
            
            imsize:tuple=(256,256),
            patchsize:int=64,
            learning_rates:list=[0.03, 0.01],
            epochs:list=[10, 100],
            batch_size:int=64,
            min_dataset_length:int=1500) -> None:
        super().__init__(subject, dataset_directory, model_input_directory, output_directory, seed)
        
        self.imsize = imsize
        self.patchsize = patchsize
        self.learning_rates = learning_rates if len(learning_rates)==2 else [learning_rates[0], learning_rates[0]]
        self.epochs = epochs if len(epochs)==2 else [epochs[0], epochs[0]]
        self.batch_size = batch_size
        self.min_dataset_length = min_dataset_length
    
    
    def setup_dataset(self):
        self.artificial = PretextTaskDatamodule(
            self.subject,
            self.dataset_directory,
            imsize=self.imsize,
            batch_size=self.batch_size,
            seed=self.seed,
            min_dataset_length=self.min_dataset_length,
            duplication=True,
            patch_localization=True,
            patch_size=self.patchsize
        )
        self.artificial.setup()
        
    
    def __setup_trainer(self, stopping_threshold:float, epochs:int, min_epochs:int, log_dir:str):
        cb = MetricTracker()
        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            stopping_threshold=stopping_threshold,
            mode='max',
            patience=5
        )
        mc = ModelCheckpoint(
            dirpath=log_dir, 
            filename='best_model_so_far',
            save_top_k=1, 
            monitor="val_loss", 
            mode='min',
            every_n_epochs=5)
        trainer = pl.Trainer(
            default_root_dir=log_dir,
            callbacks= [cb, mc, early_stopping],
            precision=16,
            benchmark=True,
            accelerator='auto', 
            devices=1, 
            max_epochs=epochs,
            min_epochs=min_epochs,
            check_val_every_n_epoch=1)
        return trainer, cb


    def execute(self, checkpoint_name:str='best_model.ckpt'):
        peranet = PeraNet()
        peranet.compile(
            learning_rate=self.learning_rates[0],
            epochs=self.epochs[0]
        )
        peranet.freeze_net(['backbone'])
        trainer, cb = self.__setup_trainer(0.75, self.epochs[0], min_epochs=3, log_dir=self.output_directory+'logs/')
        print('>>> TRAINING LATENT SPACE')
        trainer.fit(peranet, datamodule=self.artificial)
        plot_history(cb.log_metrics, self.output_directory, mode='training')
        peranet.clear_memory_bank()
        
        peranet.lr = self.learning_rates[1]
        peranet.num_epochs = self.epochs[1]
        peranet.unfreeze()
        trainer, cb = self.__setup_trainer(0.995, self.epochs[1], min_epochs=3, log_dir=self.output_directory+'logs/')
        print('>>> TRAINING WHOLE NETWORK')
        trainer.fit(peranet, datamodule=self.artificial)
        plot_history(cb.log_metrics, self.output_directory, mode='fine_tune')
        trainer.save_checkpoint(self.output_directory+checkpoint_name)
        
        
class Evaluator(__BaseClassWrapper):
    def __init__(
            self, 
            subject: str, 
            dataset_directory: str, 
            model_input_directory: str = None, 
            output_directory: str = './', 
            seed: int = 0,
            
            imsize:tuple=(256,256),
            patchsize:int=32,
            stride:int=4,
            batchsize:int=128) -> None:
        super().__init__(subject, dataset_directory, model_input_directory, output_directory, seed)
        
        self.imsize = imsize
        self.patchsize = patchsize
        self.stride = stride
        self.batchsize = batchsize
        
        # metrics
        self.roc = -1
        self.auroc = -1
        self.pro = -1
        self.aupro = -1
    
    
    def _compute_aupro(self, targets:Tensor, scores:Tensor):
        # target and scores are both a 2d map
        all_fprs, all_pros = mtr.compute_pro(
            anomaly_maps=np.array(scores),
            ground_truth_maps=np.array(targets))
        aupro = mtr.compute_aupro(all_fprs, all_pros, 0.3)
        return all_fprs, all_pros, aupro
    
    
    def _compute_f1(self, targets:Tensor, scores:Tensor):
        # target and scores are both a flat vector
        return mtr.compute_f1(targets, scores)
    
        
    def _compute_auroc(self, targets:Tensor, scores:Tensor):
        # target and scores are both a flat vector
        fpr, tpr, _ = mtr.compute_roc(targets, scores)
        auroc = mtr.compute_auc(fpr, tpr)
        return fpr, tpr, auroc
    
    
    def _get_detector_good_embeddings(self):
        model:PeraNet = PeraNet().load_from_checkpoint(self.model_dir+self.model_name)
        model.enable_mvtec_inference()
        model.eval()
        if torch.cuda.is_available():
            model.to('cuda')
        sample_imgs =  5
        if sample_imgs > len(self.mvtec_datamodule.train_dataloader().dataset.images_filenames):
            sample_imgs = len(self.mvtec_datamodule.train_dataloader().dataset.images_filenames)
        tot_embeddings = []
        print()
        pbar2 = tqdm(range(sample_imgs), desc='train data', position=1, leave=False)
        for i in pbar2:
            x, _, _ = self.mvtec_datamodule.train_dataloader().dataset.__getitem__(i)
            x_patches = extract_patches(x.unsqueeze(0), self.patch_dim, self.stride)
            if torch.cuda.is_available():
                with torch.no_grad():
                    output = model(x_patches.to('cuda'))
            else:
                with torch.no_grad():
                    output = model(x_patches)
            patches_embeddings = output['latent_space'].to('cpu')
            y_hats = get_prediction_class(output['classifier']).to('cpu')
            for i in range(len(patches_embeddings)):
                if y_hats[i] == 0:
                    tot_embeddings.append(np.array(patches_embeddings[i]))
        tot_embeddings = torch.tensor(np.array(tot_embeddings))
        return tot_embeddings
        
    
    def setup_dataset(self):
        self.artificial = PretextTaskDatamodule(
            self.subject,
            self.dataset_directory,
            imsize=self.imsize,
            batch_size=self.batchsize,
            seed=self.seed,
            min_dataset_length=500,
            duplication=True,
            patch_localization=True,
            patch_size=self.patchsize
        )
        self.artificial.setup()
        
        self.mvtec = MVTecDatamodule(
            self.dataset_directory,
            self.imsize,
            self.batchsize,
            self.seed
        )
        self.mvtec.setup()
        
    
    def setup_model(self):
        self.peranet:PeraNet = PeraNet.load_from_checkpoint(self.model_input_directory)
        self.peranet.eval()
        self.detector = AnomalyDetector()
        if self.peranet.memory_bank.shape[0] > 0:
            self.detector.fit(self.peranet.memory_bank.detach())
        else:
            self.detector.fit(self._get_detector_good_embeddings())
    
    
    def execute(self):
        if torch.cuda.is_available():
            self.peranet.to('cuda')
        # lists
        mvtec_y_true_pixel = torch.tensor([])
        mvtec_y_hat = torch.tensor([])
        anomaly_scores_pixel = torch.tensor([])
        ground_truth_maps = torch.tensor([])
        anomaly_maps = torch.tensor([])
        
        images, gts, originals = next(iter(self.mvtec_datamodule.test_dataloader()))
        j = len(images)
        # inferencing over images
        print()
        pbar3 = tqdm(range(j), desc='test images', position=2, leave=False)
        for i in pbar3:
            x_prime, gt, x = images[i], gts[i], originals[i]
            # get patches
            patches = extract_patches(x_prime[None, :], self.patch_dim, self.stride)
            if torch.cuda.is_available():
                patches = patches.to('cuda')
            with torch.no_grad():
                outputs = self.peranet(patches)
            embeddings = outputs['latent_space'].to('cpu')
            # scores
            scores = self.detector.predict(embeddings)
            # saliency map
            dim = int(np.sqrt(embeddings.shape[0]))
            saliency_map = torch.reshape(scores, (dim, dim))
            ksize = 7
            saliency_map = functional.gaussian_blur(saliency_map[None,:], kernel_size=ksize).squeeze()
            saliency_map = F.relu(saliency_map)
            saliency_map = F.interpolate(saliency_map[None,None,:], self.imsize[0], mode='bilinear').squeeze()
            # data
            flat_map = torch.nan_to_num(saliency_map.flatten(0, -1))
            flat_gt = torch.nan_to_num(gt.flatten(0, -1))
            anomaly_scores_pixel = torch.cat([anomaly_scores_pixel, flat_map])
            mvtec_y_true_pixel = torch.cat([mvtec_y_true_pixel, flat_gt])
            ground_truth_maps = torch.cat([ground_truth_maps, gt.squeeze()[None,:]])
            anomaly_maps = torch.cat([anomaly_maps, saliency_map[None,:]])

        # pixel level roc
        fpr, tpr, auc_score = self._compute_auroc(mvtec_y_true_pixel, anomaly_scores_pixel)
        self.auroc = auc_score
        # pro
        all_fprs, all_pros, au_pro = self._compute_aupro(ground_truth_maps, anomaly_maps)
        self.aupro = au_pro
        
        vis.plot_curve(
            fpr, tpr, 
            auc_score, 
            saving_path=self.output_directory,
            title='Roc curve for '+self.subject.upper()+' ['+str(self.seed)+']',
            name='pixel_roc.png')
        vis.plot_curve(
            all_fprs,
            all_pros,
            au_pro,
            saving_path=self.output_directory,
            title='Pro curve for '+self.subject.upper()+' ['+str(self.seed)+']',
            name='pro.png'
        )
        

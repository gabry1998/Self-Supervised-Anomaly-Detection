import shutil
from self_supervised.gradcam import GradCam
from self_supervised.model import GDE, AnomalyDetector, CosineEstimator, PeraNet
from self_supervised.datasets import MVTecDatamodule, GenerativeDatamodule
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import functional
from torch import Tensor
from skimage.segmentation import slic
from skimage import color
import self_supervised.support.constants as CONST
from self_supervised.support.functional import \
    extract_patches, get_all_subject_experiments, get_prediction_class, gt2label, heatmap2mask, multiclass2binary, normalize
import self_supervised.support.visualization as vis
import self_supervised.metrics as mtr
import time
import random
import numpy as np

import pytorch_lightning as pl
import os
import torch.nn.functional as F
import torch



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
        self.outputs_dir = root_output_dir+subject+'/'+mode+'/'
        
        self.model_name = model_name
        self.patch_localization = patch_localization
        self.patch_dim = patch_dim
        self.stride = stride
        self.threshold = threshold
        
        self.seed = seed
        
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
            batch_size=batch_size
        )
        self.mvtec_datamodule.setup()
    
    
    def _get_detector_good_embeddings(self):
        model:PeraNet = PeraNet.load_from_checkpoint(self.model_dir+self.model_name)
        model.enable_mvtec_inference()
        model.eval()
        if torch.cuda.is_available():
            model.to('cuda')
        sample_imgs =  5
        if sample_imgs > len(self.mvtec_datamodule.train_dataloader().dataset.images_filenames):
            sample_imgs = len(self.mvtec_datamodule.train_dataloader().dataset.images_filenames)
        tot_embeddings = []
        pbar = tqdm(range(sample_imgs), desc='train data')
        for i in pbar:
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
    
    
    def evaluate(self):
        peranet:PeraNet = PeraNet.load_from_checkpoint(
            self.model_dir+self.model_name
        )
        peranet.eval()
        if not self.patch_localization:
            tester = pl.Trainer(accelerator='auto', devices=1)
            # artificial dataset inference
            artificial_predictions = tester.predict(peranet, self.artificial_datamodule)[0]
            artificial_y_hat = artificial_predictions['y_hat']
            artificial_y_true = artificial_predictions['y_true']
            artificial_embeddings = artificial_predictions['embeddings']
            artificial_tsne = artificial_predictions['y_hat_tsne']
            
            # mvtec inference
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
            anomaly_maps = torch.tensor([])
            for i in range(len(len(artificial_tsne))):
                if mvtec_y_hat[i] == 0:
                    saliency_map = torch.zeros(self.imsize)[None, :]
                else:
                    x = mvtec_x[i]
                    # saliency is a 1x1xHxW tensor here
                    saliency_map = gradcam(x[None, :], mvtec_y_hat[i])

                anomaly_maps = torch.cat([anomaly_maps, saliency_map]) # saliency is a 1xHxW tensor here
            ground_truth_maps = mvtec_groundtruths.squeeze()
            anomaly_scores_pixel = torch.nan_to_num(saliency_map.flatten(0, -1))
            mvtec_y_true_pixel = torch.nan_to_num(gt.flatten(0, -1))
        else:
            if torch.cuda.is_available():
                peranet.to('cuda')
            # lists
            mvtec_y_true_pixel = torch.tensor([])
            mvtec_y_hat = torch.tensor([])
            anomaly_scores_pixel = torch.tensor([])
            ground_truth_maps = torch.tensor([])
            anomaly_maps = torch.tensor([])
            # setting up anomaly detector
            self.detector = AnomalyDetector()
            self.detector.fit(peranet.memory_bank.detach())
            images, gts, originals = next(iter(self.mvtec_datamodule.test_dataloader()))
            j = len(images)
            # inferencing over images
            for i in tqdm(range(j), desc='test images'):
                x_prime, gt, x = images[i], gts[i], originals[i]
                # get patches
                patches = extract_patches(x_prime[None, :], self.patch_dim, self.stride)
                if torch.cuda.is_available():
                    patches = patches.to('cuda')
                with torch.no_grad():
                    outputs = peranet(patches)
                embeddings = outputs['latent_space'].to('cpu')
                # scores
                scores = self.detector.predict(embeddings)
                # saliency map
                dim = int(np.sqrt(embeddings.shape[0]))
                saliency_map = torch.reshape(scores, (dim, dim))
                ksize = 5
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
                
        # computing metrics
        # image level roc
        if not self.patch_localization:
            fpr_image, tpr_image, auc_score_image = self._compute_auroc(mvtec_y_true, anomaly_scores)
        # pixel level roc
        fpr, tpr, auc_score = self._compute_auroc(mvtec_y_true_pixel, anomaly_scores_pixel)
        # pro
        all_fprs, all_pros, au_pro = self._compute_aupro(ground_truth_maps, anomaly_maps)
        vis.plot_curve(
            fpr, tpr, 
            auc_score, 
            saving_path=self.outputs_dir,
            title='Roc curve for '+self.subject.upper()+' ['+str(self.seed)+']',
            name='pixel_roc.png')
        vis.plot_curve(
            all_fprs,
            all_pros,
            au_pro,
            saving_path=self.outputs_dir,
            title='Pro curve for '+self.subject.upper()+' ['+str(seed)+']',
            name='pro.png'
        )

        

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
    
    
if __name__ == "__main__":
    dataset_dir='dataset/'
    root_inputs_dir='brutta_brutta_copia/computations/'
    root_outputs_dir='brutta_brutta_copia/computations/'
    imsize=(256,256)
    seed=204110176
    patch_localization=True
    
    evaluator = Evaluator(
        dataset_dir=dataset_dir,
        root_input_dir=root_inputs_dir,
        root_output_dir=root_outputs_dir,
        subject='bottle',
        model_name='best_model.ckpt',
        patch_localization=patch_localization,
        patch_dim=32,
        stride=8,
        seed=204110176
    )
    evaluator.setup_dataset()
    evaluator.evaluate()
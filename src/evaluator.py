import shutil
from self_supervised.gradcam import GradCam
from self_supervised.models import AnomalyDetector, PeraNet
from self_supervised.datasets import MVTecDatamodule, PretextTaskDatamodule
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import functional
from torch import Tensor
from skimage.segmentation import slic
from skimage import color
from sklearn.metrics import classification_report
from torchmetrics import PrecisionRecallCurve, JaccardIndex
from self_supervised.functional import \
    extract_patches, get_all_subject_experiments, get_prediction_class, normalize
from self_supervised.converters import \
    gt2label, heatmap2mask, multiclass2binary
import self_supervised.visualization as vis
from sklearn.metrics import PrecisionRecallDisplay
import self_supervised.metrics as mtr
import matplotlib.pyplot as plt
import time
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import os
import torch.nn.functional as F
import torch


class ArtificialEvaluator:
    def __init__(
            self,
            dataset_dir:str=None,
            root_input_dir:str=None, 
            root_output_dir:str=None,
            subject:str=None,
            model_name:str='best_model.ckpt',
            seed=0) -> None:
        
        self.dataset_dir = dataset_dir
        self.model_dir = root_input_dir+subject+'/'
        self.subject = subject
        self.root_input_dir = root_input_dir,
        self.root_output_dir = root_output_dir
        self.outputs_dir = root_output_dir+subject+'/'
        
        self.model_name = model_name
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.f1 = -1
        self.f1_good = -1
        self.f1_poly = -1
        self.f1_rect = -1
        self.f1_line = -1
        
        
    def setup_dataset(self, imsize:tuple=(256,256), artificial_batch_size:int=96, mvtec_batch_size:int=128):
        
        self.imsize = imsize
        
        self.artificial_datamodule = PretextTaskDatamodule(
        self.subject,
        self.dataset_dir+self.subject+'/',
        imsize=imsize,
        batch_size=artificial_batch_size,
        seed=self.seed,
        duplication=True,
        min_dataset_length=500,
        patch_localization=False
        )
        self.artificial_datamodule.setup('test')    
        
    def evaluate(self):
        if os.path.exists('lightning_logs/'):
            shutil.rmtree('lightning_logs/')
        peranet:PeraNet = PeraNet.load_from_checkpoint(
            self.model_dir+self.model_name
        )
        peranet.eval()
        tester = pl.Trainer(accelerator='auto', devices=1)
        # artificial dataset inference
        artificial_predictions = tester.predict(peranet, self.artificial_datamodule)[0]
        artificial_y_hat = artificial_predictions['y_hat']
        artificial_y_true = artificial_predictions['y_true']
        artificial_embeddings = artificial_predictions['embedding']
        
        report = classification_report(
                np.array(artificial_y_true), 
                np.array(artificial_y_hat),
                labels=[0,1,2,3],
                target_names=['good','polygon','rectangles','line'],
                digits=4,
                output_dict=True)
        #print(report)
        report = mtr.metrics_to_dataframe(report)
        mtr.export_dataframe(report, self.outputs_dir, self.subject+'_artificial_report.csv')
        self.f1 = report['accuracy']['f1-score']
        self.f1_good = report['good']['f1-score']
        self.f1_poly = report['polygon']['f1-score']
        self.f1_rect = report['rectangles']['f1-score']
        self.f1_line = report['line']['f1-score']
        #print(report['good']['precision'])
        # computing anomaly scores
        detector = AnomalyDetector()
        detector.fit(peranet.memory_bank)
        anomaly_scores = detector.predict(artificial_embeddings)
        y = multiclass2binary(artificial_y_true)
        fpr_image, tpr_image, auc_score_image = self._compute_auroc(y, anomaly_scores)
        
        self.image_auroc = auc_score_image
        vis.plot_curve(
            fpr_image, tpr_image, 
            auc_score_image, 
            saving_path=self.outputs_dir,
            title='Roc curve for '+self.subject.upper()+' ['+str(self.seed)+']',
            name='artificial_classification_roc.png')
        
    def _compute_auroc(self, targets:Tensor, scores:Tensor):
        # target and scores are both a flat vector
        fpr, tpr, _ = mtr.compute_roc(targets, scores)
        auc_score = mtr.compute_auc(fpr, tpr)
        return fpr, tpr, auc_score    


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
        
        self.dataset_dir = dataset_dir
        self.model_dir = root_input_dir+subject+'/'
        self.subject = subject
        self.root_input_dir = root_input_dir,
        self.root_output_dir = root_output_dir
        self.outputs_dir = root_output_dir+subject+'/'
        
        self.model_name = model_name
        self.patch_localization = patch_localization
        self.patch_dim = patch_dim
        self.stride = stride
        self.threshold = threshold
        
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        # scores
        self.artificial_f1 = None
        self.image_auroc = None
        self.pixel_auroc = None
        self.aupro = None
        self.iou = None
        self.f1 = None
        self.fpr_image = None
        self.tpr_image = None
        self.fpr = None
        self.tpr = None
        self.all_fprs = None
        self.all_pros = None
        
        
        
    def setup_dataset(self, imsize:tuple=(256,256), artificial_batch_size:int=96, mvtec_batch_size:int=128):
        
        self.imsize = imsize
        
        self.artificial_datamodule = PretextTaskDatamodule(
        self.subject,
        self.dataset_dir+self.subject+'/',
        imsize=imsize,
        batch_size=artificial_batch_size,
        seed=self.seed,
        duplication=True,
        min_dataset_length=500,
        patch_localization=False
        )
        self.artificial_datamodule.setup('test')
        
        
        self.mvtec_datamodule = MVTecDatamodule(
            root_dir=self.dataset_dir+self.subject+'/',
            imsize=imsize,
            batch_size=mvtec_batch_size
        )
        self.mvtec_datamodule.setup()
    
    
    def _get_detector_good_embeddings(self):
        model:PeraNet = PeraNet().load_from_checkpoint(self.model_dir+self.model_name)
        model.enable_mvtec_inference()
        model.eval()
        if torch.cuda.is_available():
            model.to('cuda')
        sample_imgs =  1
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
            #y_hats = get_prediction_class(output['classifier']).to('cpu')
            for i in range(len(patches_embeddings)):
            #    if y_hats[i] == 0:
             #       tot_embeddings.append(np.array(patches_embeddings[i]))
                tot_embeddings.append(np.array(patches_embeddings[i]))
        tot_embeddings = torch.tensor(np.array(tot_embeddings))
        return tot_embeddings
    
    
    def evaluate(self):
        if os.path.exists('lightning_logs/'):
            shutil.rmtree('lightning_logs/')
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
            artificial_embeddings = artificial_predictions['embedding']
            artificial_tsne = artificial_predictions['y_tsne']
        
            
            # mvtec inference
            peranet.enable_mvtec_inference()
            mvtec_predictions = tester.predict(peranet, self.mvtec_datamodule)[0]
            mvtec_x = mvtec_predictions['x_prime']
            mvtec_y_hat = mvtec_predictions['y_hat']
            mvtec_y_true = mvtec_predictions['y_true']
            mvtec_groundtruths = mvtec_predictions['groundtruth']
            mvtec_embeddings = mvtec_predictions['embedding']
            mvtec_tsne = mvtec_predictions['y_tsne']
            
            
            # normalization
            artificial_embeddings = torch.nn.functional.normalize(artificial_embeddings, p=2, dim=1)
            mvtec_embeddings = torch.nn.functional.normalize(mvtec_embeddings, p=2, dim=1)
            
            # computing anomaly scores
            detector = AnomalyDetector()
            detector.fit(peranet.memory_bank)
            anomaly_scores = detector.predict(mvtec_embeddings)
            
            # gradcam for pixel metrics and pro
            gradcam = GradCam(model=peranet)
            anomaly_maps = torch.tensor([])
            for i in range(len(mvtec_x)):
                if mvtec_y_hat[i] == 0:
                    # saliency is a 1x1xHxW tensor here
                    saliency_map = torch.zeros(self.imsize)[None, None, :]
                else:
                    x = mvtec_x[i]
                    # saliency is a 1x1xHxW tensor here
                    saliency_map = gradcam(x[None, :], mvtec_y_hat[i])
                anomaly_maps = torch.cat([anomaly_maps, saliency_map]) # saliency is a 1xHxW tensor here
            anomaly_scores_pixel = torch.nan_to_num(anomaly_maps.flatten(0, -1))
            ground_truth_maps = mvtec_groundtruths.squeeze()
            anomaly_maps = anomaly_maps.squeeze()
            mvtec_y_true_pixel = torch.nan_to_num(ground_truth_maps.flatten(0, -1))
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
            #if peranet.memory_bank.shape[0] > 0:
            #    self.detector.fit(peranet.memory_bank.detach())
            #else:
            self.detector.fit(self._get_detector_good_embeddings())
            # inferencing over mvtec images 
            images, gts, _ = next(iter(self.mvtec_datamodule.test_dataloader()))
            j = len(images)
            print()
            pbar3 = tqdm(range(j), desc='mvtec test images', position=2, leave=False)
            for i in pbar3:
                x_prime, gt = images[i], gts[i]
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
                ksize = 7
                saliency_map = functional.gaussian_blur(saliency_map[None,:], kernel_size=ksize).squeeze()
                saliency_map = F.relu(saliency_map)
                saliency_map = F.interpolate(saliency_map[None,None,:], self.imsize[0], mode='bilinear').squeeze()
                
                #print(min(saliency_map),max(saliency_map))
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
            self.fpr_image, self.tpr_image, auc_score_image, t = self._compute_auroc(mvtec_y_true, anomaly_scores)
            self.image_auroc = auc_score_image
            vis.plot_curve(
                self.fpr_image, self.tpr_image, 
                auc_score_image,
                saving_path=self.outputs_dir,
                title='Roc curve for '+self.subject.upper(),
                name=self.subject+'_image_roc.png')
            self.f1 = self._compute_f1(mvtec_y_true, anomaly_scores, t)
            vis.plot_tsne(
                torch.cat([artificial_embeddings, mvtec_embeddings]),
                torch.cat([artificial_tsne, mvtec_tsne]),
                saving_path=self.outputs_dir,
                title=self.subject.upper()+' feature visualization',
                name=self.subject+'_tsne.png')
        else:
            # pixel level roc
            self.fpr, self.tpr, auc_score, px_t = self._compute_auroc(mvtec_y_true_pixel, anomaly_scores_pixel)
            self.pixel_auroc = auc_score
            self.iou = self._compute_iou(mvtec_y_true_pixel, anomaly_scores_pixel, px_t)
            # pro
            print(ground_truth_maps.shape, anomaly_maps.shape)
            self.all_fprs, self.all_pros, au_pro = self._compute_aupro(ground_truth_maps, anomaly_maps)
            self.aupro = au_pro
            vis.plot_curve(
                self.fpr, self.tpr, 
                auc_score,
                saving_path=self.outputs_dir,
                title='Roc curve for '+self.subject.upper(),
                name=self.subject+'_pixel_roc.png')
            vis.plot_curve(
                self.all_fprs,
                self.all_pros,
                au_pro,
                saving_path=self.outputs_dir,
                title='Pro curve for '+self.subject.upper(),
                name=self.subject+'_pro.png'
            )
        
        
    def _compute_iou(self, target, scores, t):
        target = torch.where(target>0, 1, 0)
        metric = JaccardIndex(2,threshold=t)
        iou = metric(scores, target)
        return iou.item()
    
    def _compute_aupro(self, targets:Tensor, scores:Tensor):
        # target and scores are both a 2d map
        all_fprs, all_pros = mtr.compute_pro(
            anomaly_maps=np.array(scores),
            ground_truth_maps=np.array(targets))
        au_pro = mtr.compute_aupro(all_fprs, all_pros, 0.3)
        return all_fprs, all_pros, au_pro
    
    
    def _compute_f1(self, targets:Tensor, scores:Tensor, t):
        # target and scores are both a flat vector
        return mtr.compute_f1(targets, scores, t)
           
    def _compute_auroc(self, targets:Tensor, scores:Tensor):
        # target and scores are both a flat vector
        fpr, tpr, _ = mtr.compute_roc(targets, scores)
        precision_recall_curve = PrecisionRecallCurve()
        precision, recall, t = precision_recall_curve(scores, targets)
        f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
        print(f1_score)
        value = t[np.argmax(f1_score)]
        value
        
        auc_score = mtr.compute_auc(fpr, tpr)
        return fpr, tpr, auc_score, value


def get_textures_names():
    return ['carpet','grid','leather','tile','wood']


def obj_set_one():
    return [
        'bottle',
        'cable',
        'capsule',
        'hazelnut',
        'metal_nut']


def obj_set_two():
    return [
        'pill',
        'screw',
        'toothbrush',
        'transistor',
        'zipper']


def evaluate(
        dataset_dir='dataset/',
        root_inputs_dir='brutta_brutta_copia/computations/',
        root_outputs_dir='brutta_brutta_copia/computations/',
        imsize=(256,256),
        patch_dim = 32,
        stride=8,
        seed=123456789,
        patch_localization=True,
        experiments_list=[],
        artificial_batch_size=256,
        tables_output:str=None
        ):
    
    image_aurocs = []
    pixel_aurocs = []
    iou = []
    f1 = []
    aupros = []
    
    objects_names = []
    textures_names = []
    
    roc_curves = []
    pixel_roc_curves = []
    per_region_overlaps = []
    text_roc_curves = []
    text_pixel_roc_curves = []
    text_per_region_overlaps = []
    
    pbar = tqdm(range(len(experiments_list)), position=0, leave=False)
    for i in pbar:
        pbar.set_description('Evaluation pipeline | current subject is '+experiments_list[i].upper())
        evaluator = Evaluator(
            dataset_dir=dataset_dir,
            root_input_dir=root_inputs_dir,
            root_output_dir=root_outputs_dir,
            subject=experiments_list[i],
            model_name='best_model.ckpt',
            patch_localization=patch_localization,
            patch_dim=patch_dim,
            stride=stride,
            seed=seed
        )
        evaluator.setup_dataset(artificial_batch_size=artificial_batch_size)
        evaluator.evaluate()
        if not experiments_list[i] in np.array(['carpet','grid','leather','tile','wood']):
            objects_names.append(experiments_list[i])
            roc_curves.append((evaluator.fpr_image, evaluator.tpr_image))
            pixel_roc_curves.append((evaluator.fpr, evaluator.tpr))
            per_region_overlaps.append((evaluator.all_fprs, evaluator.all_pros))
        else:
            textures_names.append(experiments_list[i])
            text_roc_curves.append((evaluator.fpr_image, evaluator.tpr_image))
            text_pixel_roc_curves.append((evaluator.fpr, evaluator.tpr))
            text_per_region_overlaps.append((evaluator.all_fprs, evaluator.all_pros))
        image_aurocs.append(evaluator.image_auroc)
        pixel_aurocs.append(evaluator.pixel_auroc)
        f1.append(evaluator.f1)
        iou.append(evaluator.iou)
        aupros.append(evaluator.aupro)
        
        os.system('clear')
    
    if len(experiments_list) > 0:
        metric_dict={}
        if not patch_localization:
            metric_dict['AUC (image)'] = np.append(
                image_aurocs, 
                np.mean(image_aurocs))
            metric_dict['F1 (image)'] = np.append(
                f1, 
                np.mean(f1))
            if len(objects_names) > 0:
                vis.plot_multiple_curve(roc_curves=roc_curves,names=objects_names, saving_path=tables_output, name='object_rocs.png')
            if len(textures_names) > 0:
                vis.plot_multiple_curve(roc_curves=text_roc_curves,names=textures_names, saving_path=tables_output, name='textures_rocs.png')
        else:
            metric_dict['AUC (pixel)'] = np.append(
                pixel_aurocs, 
                np.mean(pixel_aurocs))
            metric_dict['IOU'] = np.append(
                iou, 
                np.mean(iou))
            metric_dict['AUPRO'] = np.append(
                aupros, 
                np.mean(aupros))
            if len(textures_names) > 0:
                vis.plot_multiple_curve(roc_curves=text_pixel_roc_curves, names=textures_names, saving_path=tables_output, name='textures_pixel_rocs.png')
                vis.plot_multiple_curve(roc_curves=text_per_region_overlaps, names=textures_names, saving_path=tables_output, name='textures_pixel_pros.png')
            if len(objects_names) > 0:
                vis.plot_multiple_curve(roc_curves=per_region_overlaps, names=objects_names, saving_path=tables_output, name='object_pixel_pros.png')
                vis.plot_multiple_curve(roc_curves=pixel_roc_curves, names=objects_names, saving_path=tables_output, name='object_pixel_rocs.png')
            
        scores = mtr.metrics_to_dataframe(metric_dict, np.array(experiments_list+['average']))
        if patch_localization:
            mtr.export_dataframe(scores, saving_path=tables_output+'csv/', name='patch_all_scores.csv')
            mtr.export_dataframe(scores, saving_path=tables_output+'latex/', name='patch_all_scores.tex', mode='latex')
            mtr.export_dataframe(scores, saving_path=tables_output+'markdown/', name='patch_all_scores.md', mode='markdown')
        else:
            mtr.export_dataframe(scores, saving_path=tables_output+'csv/', name='image_all_scores.csv')
            mtr.export_dataframe(scores, saving_path=tables_output+'latex/', name='image_all_scores.tex', mode='latex')
            mtr.export_dataframe(scores, saving_path=tables_output+'markdown/', name='image_all_scores.md', mode='markdown')
    textures_scores = None
    if len(textures_names) > 0:
        textures_scores:pd.DataFrame = scores.loc[textures_names]
        textures_avg = textures_scores.mean(axis='index').to_dict()
        textures_avg = pd.DataFrame(textures_avg, columns=textures_scores.keys(), index=['average'])
        textures_scores = pd.concat([textures_scores, textures_avg])
        if patch_localization:
            mtr.export_dataframe(textures_scores, saving_path=tables_output+'csv/', name='patch_textures_scores.csv')
            mtr.export_dataframe(textures_scores, saving_path=tables_output+'latex/', name='patch_textures_scores.tex', mode='latex')
            mtr.export_dataframe(textures_scores, saving_path=tables_output+'markdown/', name='patch_textures_scores.md', mode='markdown')
        else:
            mtr.export_dataframe(textures_scores, saving_path=tables_output+'csv/', name='image_textures_scores.csv')
            mtr.export_dataframe(textures_scores, saving_path=tables_output+'latex/', name='image_textures_scores.tex', mode='latex')
            mtr.export_dataframe(textures_scores, saving_path=tables_output+'markdown/', name='image_textures_scores.md', mode='markdown')
    objects_scores = None
    if len(objects_names) > 0:
        objects_scores:pd.DataFrame = scores.loc[objects_names]
        objects_avg = objects_scores.mean(axis='index').to_dict()
        objects_avg = pd.DataFrame(objects_avg, columns=objects_avg.keys(), index=['average'])
        objects_scores = pd.concat([objects_scores, objects_avg])
        if patch_localization:
            mtr.export_dataframe(objects_scores, saving_path=tables_output+'csv/', name='patch_objects_scores.csv')
            mtr.export_dataframe(objects_scores, saving_path=tables_output+'latex/', name='patch_objects_scores.tex', mode='latex')
            mtr.export_dataframe(objects_scores, saving_path=tables_output+'markdown/', name='patch_objects_scores.md', mode='markdown')
        else:
            mtr.export_dataframe(objects_scores, saving_path=tables_output+'csv/', name='image_objects_scores.csv')
            mtr.export_dataframe(objects_scores, saving_path=tables_output+'latex/', name='image_objects_scores.tex', mode='latex')
            mtr.export_dataframe(objects_scores, saving_path=tables_output+'markdown/', name='imageh_objects_scores.md', mode='markdown')
        
    return scores, textures_scores, objects_scores  
    
    


def evaluate_artificial(
        dataset_dir='dataset/',
        root_inputs_dir='brutta_brutta_copia/computations/',
        root_outputs_dir='brutta_brutta_copia/computations/',
        imsize=(256,256),
        seed=123456789,
        experiments_list=[],
        artificial_batch_size=256,
        tables_output='brutta_copia/'
        ):
    auc = []
    f1 = []
    f1_good = []
    f1_poly = []
    f1_rect = []
    f1_line = []
    objects_names = []
    textures_names = []
    
    pbar = tqdm(range(len(experiments_list)), position=0, leave=False)
    for i in pbar:
        pbar.set_description('Evaluation pipeline | current subject is '+experiments_list[i].upper())
        evaluator = ArtificialEvaluator(
            dataset_dir=dataset_dir,
            root_input_dir=root_inputs_dir,
            root_output_dir=root_outputs_dir,
            subject=experiments_list[i],
            model_name='best_model.ckpt',
            seed=seed
        )
        evaluator.setup_dataset(artificial_batch_size=artificial_batch_size)
        evaluator.evaluate()
        if not experiments_list[i] in np.array(['carpet','grid','leather','tile','wood']):
            objects_names.append(experiments_list[i])
        else:
            textures_names.append(experiments_list[i])
        auc.append(evaluator.image_auroc)
        f1.append(evaluator.f1)
        f1_good.append(evaluator.f1_good)
        f1_poly.append(evaluator.f1_poly)
        f1_rect.append(evaluator.f1_rect)
        f1_line.append(evaluator.f1_line)
    
    if len(experiments_list) > 0:
        metric_dict={}
        metric_dict['AUC'] = np.append(
            auc, 
            np.mean(auc))
        metric_dict['Accuracy'] = np.append(
            f1, 
            np.mean(f1))
        metric_dict['F1 good'] = np.append(
            f1_good, 
            np.mean(f1_good))
        metric_dict['F1 polygons'] = np.append(
            f1_poly, 
            np.mean(f1_poly))
        metric_dict['F1 rectangles'] = np.append(
            f1_rect, 
            np.mean(f1_rect))
        metric_dict['F1 line'] = np.append(
            f1_line, 
            np.mean(f1_line))
        scores = mtr.metrics_to_dataframe(metric_dict, np.array(experiments_list+['average']))
        mtr.export_dataframe(scores, saving_path=tables_output+'csv/', name='artificial_all_scores.csv')
        mtr.export_dataframe(scores, saving_path=tables_output+'latex/', name='artificial_all_scores.tex', mode='latex')
    textures_scores = None
    if len(textures_names) > 0:
        textures_scores:pd.DataFrame = scores.loc[textures_names]
        textures_avg = textures_scores.mean(axis='index').to_dict()
        textures_avg = pd.DataFrame(textures_avg, columns=textures_scores.keys(), index=['average'])
        textures_scores = pd.concat([textures_scores, textures_avg])

        mtr.export_dataframe(textures_scores, saving_path=tables_output+'csv/', name='artificial_textures_scores.csv')
        mtr.export_dataframe(textures_scores, saving_path=tables_output+'latex/', name='artificial_textures_scores.tex', mode='latex')
    objects_scores = None
    if len(objects_names) > 0:
        objects_scores:pd.DataFrame = scores.loc[objects_names]
        objects_avg = objects_scores.mean(axis='index').to_dict()
        objects_avg = pd.DataFrame(objects_avg, columns=objects_avg.keys(), index=['average'])
        objects_scores = pd.concat([objects_scores, objects_avg])
        mtr.export_dataframe(objects_scores, saving_path=tables_output+'csv/', name='artificial_objects_scores.csv')
        mtr.export_dataframe(objects_scores, saving_path=tables_output+'latex/', name='artificial_objects_scores.tex', mode='latex')
        
    return scores, textures_scores, objects_scores  


def mvtec_eval():
    experiments = get_all_subject_experiments('dataset/')
    textures = get_textures_names()
    obj1 = obj_set_one()
    obj2 = obj_set_two()
    experiments_list = experiments
    patch_localization = False
    mode = 'patch_level' if patch_localization else 'image_level'
    
    evaluate(
        dataset_dir='dataset/',
        root_inputs_dir='brutta_copia/outputs/'+mode+'/computations/',
        root_outputs_dir='brutta_copia/b/'+mode+'/computations/',
        imsize=(256,256),
        patch_dim = 32,
        stride=8,
        seed=123456789,
        patch_localization=patch_localization,
        experiments_list=experiments_list,
        tables_output='brutta_copia/b/'+mode+'/tables/'
    )

def artificial_only():
    experiments = get_all_subject_experiments('dataset/')
    textures = get_textures_names()
    obj1 = obj_set_one()
    obj2 = obj_set_two()
    experiments_list = experiments
    mode = 'image_level'
    
    evaluate_artificial(
        dataset_dir='dataset/',
        root_inputs_dir='brutta_copia/outputs/'+mode+'/computations/',
        root_outputs_dir='brutta_copia/b/'+mode+'/computations/',
        imsize=(256,256),
        seed=123456789,
        experiments_list=experiments_list,
        tables_output='brutta_copia/b/'+mode+'/tables/'
    )
     
    
if __name__ == "__main__":
    artificial_only()
    #mvtec_eval()
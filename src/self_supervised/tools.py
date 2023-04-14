from __future__ import annotations
from torchmetrics import JaccardIndex, PrecisionRecallCurve
from tqdm import tqdm
from self_supervised.datasets import PretextTaskDatamodule, MVTecDatamodule
from self_supervised.models import PeraNet, AnomalyDetector
from self_supervised.custom_callbacks import MetricTracker
from torch import Tensor
from torch.nn.functional import softmax
from self_supervised.constants import ModelOutputsContainer, EvaluationOutputContainer, METRICS, LOCALIZATION_OUTPUTS
from self_supervised.converters import imagetensor2array, multiclass2binary
from self_supervised.functional import get_prediction_class
from self_supervised import visualization as vis
from self_supervised import metrics as mtr
from PIL import Image, ImageDraw
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.transforms import functional
from torch.backends import cudnn
from torch import autograd
import torch.nn.functional as F
import os
import shutil
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import pytorch_lightning as pl


# class for evaluating the model outputs
class Evaluator:
    def __init__(
            self, 
            evaluation_metrics:list=[]) -> None:
        
        self.evaluation_metrics = np.array(evaluation_metrics)
        self.scores = EvaluationOutputContainer() 
    
    
    def plot_tsne(
            self,
            outputs_artificial:ModelOutputsContainer,
            outputs_real:ModelOutputsContainer, 
            subject:str, 
            outputs_dir:str):
        print('>>> tsne plot')
        vis.plot_tsne(
            torch.cat([outputs_artificial.embedding_vectors, outputs_real.embedding_vectors]),
            torch.cat([outputs_artificial.y_true_multiclass_labels, outputs_real.y_true_multiclass_labels]),
            saving_path=outputs_dir,
            title=subject.upper()+' feature visualization',
            name=subject+'_tsne.png')
    
    
    def evaluate(
            self,
            output_container:ModelOutputsContainer, 
            subject:str, 
            outputs_dir:str,
            patch_level:bool=False): 
        print('>>> start evaluation')    
        
        # check valid metrics
        if self.evaluation_metrics.size == 0:
            print('No metrics selected')
        
        
        
        check_metrics = np.isin(self.evaluation_metrics, METRICS(), assume_unique=True)
        if not np.all(check_metrics):
            wrong_metrics = np.where(check_metrics==False)[0]
            print('Wrong metric(s): ', end='')
            print([self.evaluation_metrics[i] for i in wrong_metrics])
            print('Correct metrics list: ', end='')
            print(METRICS())
            exit(0)
        
        
        flat_y_true_binary_labels = output_container.y_true_binary_labels
        flat_anomaly_maps = output_container.anomaly_maps
        if patch_level:
            flat_y_true_binary_labels = output_container.ground_truths
            flat_y_true_binary_labels = torch.flatten(flat_y_true_binary_labels, 0,-1)
            flat_anomaly_maps = torch.flatten(flat_anomaly_maps, 0,-1)
            
        threshold = self._get_threshold(flat_anomaly_maps, flat_y_true_binary_labels)
        
        # computing auroc
        if 'auroc' in self.evaluation_metrics:
            print(' pixel auroc' if patch_level else '>>> image auroc')
            # y_true_binary_labels and y_hat are both a flat vector
            
            fpr, tpr, _ = mtr.compute_roc(flat_y_true_binary_labels, flat_anomaly_maps)
            auc_score = mtr.compute_auc(fpr, tpr)
            vis.plot_curve(
                fpr, tpr, 
                auc_score, 
                saving_path=outputs_dir,
                title='Roc curve for '+subject.upper(),
                name=subject+'_pixel_roc.png' if patch_level else subject+'_image_roc.png')
            self.scores.auroc = auc_score
        
        # computing image f1-score
        if 'f1-score' in self.evaluation_metrics:
            if not patch_level:
                print('\'f1-score\' not a valid metric for \'patch-level\' mode')
                exit(0)
            print(' f1-score')
            score= mtr.compute_f1(flat_y_true_binary_labels, flat_anomaly_maps, threshold)
            self.scores.f1_score = score
        
        # computing aupro
        if 'aupro' in self.evaluation_metrics:
            if not patch_level:
                print('\'aupro\' not a valid metric for \'image-level\' mode')
                exit(0)
            print(' aupro')
            all_fprs, all_pros = mtr.compute_pro(
                anomaly_maps=np.array(output_container.anomaly_maps.squeeze()),
                ground_truth_maps=np.array(output_container.ground_truths.squeeze()))
            aupro = mtr.compute_aupro(all_fprs, all_pros, 0.3)
            vis.plot_curve(
                all_fprs,
                all_pros,
                aupro,
                saving_path=outputs_dir,
                title='Pro curve for '+subject.upper(),
                name=subject+'_pro.png'
            )
            self.scores.aupro = aupro
        
        if 'iou' in self.evaluation_metrics:
            if not patch_level:
                print('\'iou\' not a valid metric for \'image-level\' mode')
                exit(0)
            print(' iou')
            target = torch.where(flat_y_true_binary_labels>0, 1, 0)
            metric = JaccardIndex(2,threshold=threshold)
            iou:Tensor = metric(flat_anomaly_maps, target)
            self.scores.iou = iou.item()
            

    
    def _get_threshold(self, scores:Tensor, targets:Tensor):
        precision_recall_curve = PrecisionRecallCurve()
        precision, recall, t = precision_recall_curve(scores, targets)
        f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
        value = t[np.argmax(f1_score)]
        return value


# class for localizing defects
class Localizer:
    def __init__(
            self,
            outputs_container:ModelOutputsContainer,
            outputs_dir:str,
            outputs_list:list[str] = ['original', 'localization']) -> None:
        self.outputs_container = outputs_container
        self.outputs_dir = outputs_dir
        self.outputs_list = np.array(outputs_list)
        
    def localize(
            self,
            threshold:int):
        
        # check valid outputs
        if self.outputs_list.size == 0:
            print('No outputs selected')
            
        check_outputs = np.isin(self.outputs_list, LOCALIZATION_OUTPUTS(), assume_unique=True)
        if not np.all(check_outputs):
            wrong_outs = np.where(check_outputs==False)[0]
            print('Wrong output(s): ', end='')
            print([self.outputs_list[i] for i in wrong_outs])
            print('Correct outputs list: ', end='')
            print(LOCALIZATION_OUTPUTS())
            exit(0)
        
        num_images = self.outputs_container.anomaly_maps.shape[0]
        pbar = tqdm(range(num_images), leave=True)
        for i in pbar:
            out_suffix = '/'+str(i)+'_img/'
            # tensors
            original_tensor_image = self.outputs_container.original_data[i]
            ground_truth = self.outputs_container.ground_truths[i]
            anomaly_map = self.outputs_container.anomaly_maps[i]
            anomaly_map_upsampled = upsample(anomaly_map[None, :])
            predicted_mask = anomaly_map_upsampled > threshold
            # numpy arrays for plot
            original_image_ndarray = imagetensor2array(original_tensor_image)
            anomaly_map_ndarray = imagetensor2array(anomaly_map, integer=False)
            anomaly_map_upsampled_ndarray = imagetensor2array(anomaly_map_upsampled[0], integer=False)
            ground_truth_ndarray = imagetensor2array(ground_truth)
            localization_ndarray = vis.apply_heatmap(original_tensor_image[None, :], anomaly_map_upsampled)
            predicted_mask_ndarray = imagetensor2array(predicted_mask[0]).squeeze()
            segmentation_ndarray = vis.apply_segmentation(original_image_ndarray, predicted_mask_ndarray)
            #print('>>> plotting images')
            if 'original' in self.outputs_list:
                pbar.set_description('plotting original image        (image %i)' % i)
                pbar.refresh()
                #print(' original')
                vis.plot_single_image(
                    img=original_image_ndarray, 
                    saving_path=self.outputs_dir+out_suffix,
                    name=str(i)+'_original.png')
            if 'ground_truth' in self.outputs_list:
                pbar.set_description('plotting ground_truth          (image %i)' % i)
                pbar.refresh()
                #print(' ground_truth')
                vis.plot_single_image(
                    img=ground_truth_ndarray, 
                    saving_path=self.outputs_dir+out_suffix,
                    name=str(i)+'_ground_truth.png')
            if 'anomaly_map' in self.outputs_list:
                pbar.set_description('plotting anomaly_map           (image %i)' % i)
                pbar.refresh()
                #print(' anomaly_map')
                vis.plot_single_image(
                    img=anomaly_map_ndarray, 
                    saving_path=self.outputs_dir+out_suffix,
                    name=str(i)+'_anomaly_map.png')
            if 'anomaly_map_upsampled' in self.outputs_list:
                pbar.set_description('plotting anomaly_map_upsampled (image %i)' % i)
                pbar.refresh()
                #print(' anomaly_map_upsampled')
                vis.plot_single_image(
                    img=anomaly_map_upsampled_ndarray, 
                    saving_path=self.outputs_dir+out_suffix,
                    name=str(i)+'_anomaly_map_upsampled.png')
            if 'localization' in self.outputs_list:
                pbar.set_description('plotting localization          (image %i)' % i)
                pbar.refresh()
                #print(' localization')
                vis.plot_single_image(
                    img=localization_ndarray, 
                    saving_path=self.outputs_dir+out_suffix,
                    name=str(i)+'_localization.png')
            if 'segmentation' in self.outputs_list:
                pbar.set_description('plotting segmentation          (image %i)' % i)
                pbar.refresh()
                #print(' segmentation')
                vis.plot_single_image(
                    img=segmentation_ndarray, 
                    saving_path=self.outputs_dir+out_suffix,
                    name=str(i)+'_segmentation.png')
        

# class for checking classification errors based on probabilities
# eg. check why good image predicted as a defect
class ErrorAnalyzer:
    def __init__(
            self,
            raw_predictions:Tensor,
            true_labels:Tensor,
            tensor_images:Tensor) -> None:
        
        self.probabilities = softmax(raw_predictions, dim=1)
        self.true_labels = true_labels
        self.tensor_images = tensor_images
        self.pretty_labels = ['good','polygon','rectangle','segment']
        
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

# function for training over a single category
def training(
        dataset_dir:str,
        outputs_dir:str,
        subject:str,
        imsize:tuple=(256,256),
        patch_localization:bool=False,
        patchsize:int=32,
        seed:int=0,
        batch_size:int=96,
        projection_training_params:tuple=(10,0.03),
        fine_tune_params:tuple=(30,0.005)
        ) -> None:
    
    print('>>> initializing training')
    autograd.anomaly_mode.set_detect_anomaly(False)
    autograd.profiler.profile(False)
    
    checkpoint_name = 'best_model.ckpt'
    proj_epochs, proj_lr = projection_training_params
    fine_tune_epochs, fine_tune_lr = fine_tune_params
    
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    if os.path.exists(outputs_dir+'logs/'):
        shutil.rmtree(outputs_dir+'logs/')
    
    print('>>> setting seeds')
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
        dataset_dir,
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
    print('>>> training projection head')
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
    print('>>> Fine tuning') 
    trainer.fit(pretext_model, datamodule=datamodule)
    trainer.save_checkpoint(outputs_dir+checkpoint_name)
    print('>>> training plot')
    vis.plot_history(cb.log_metrics, outputs_dir, mode='fine_tune')
    

# function for inferencing over a single category
def inference(
        model_input_dir:str,
        dataset_dir:str,
        subject:str,
        mvtec_inference:bool=True,
        patch_localization:bool=False,
        max_images_to_inference:int=-1
        ) -> ModelOutputsContainer:
    print('>>> initializing inference')
    if os.path.exists('lightning_logs/'):
        shutil.rmtree('lightning_logs/')
    
    print('>>> preparing model')
    model:PeraNet = PeraNet.load_from_checkpoint(model_input_dir)
    model.eval()
    if patch_localization:
        model.enable_patch_level_mode()
    
    tester = pl.Trainer(accelerator='auto', devices=1, limit_predict_batches=max_images_to_inference)
    
    print('>>> preparing datamodule')
    datamodule = None
    if mvtec_inference:
        print(' mvtec dataset')
        model.enable_mvtec_inference()
        datamodule = MVTecDatamodule(
            dataset_dir,
            batch_size=1,
        )
    else:
        print(' artificial dataset')
        datamodule = PretextTaskDatamodule(
            subject=subject,
            root_dir=dataset_dir,
            min_dataset_length=500,
            batch_size=1,
        )
    print('>>> doing prediction')
    predictions:list[ModelOutputsContainer] = tester.predict(model, datamodule=datamodule)
    output = ModelOutputsContainer()
    output.from_list(predictions)
    print('>>> anomaly detection phase')
    if patch_localization:
        b = len(predictions)
        p = model.num_patches
        detector = AnomalyDetector(patch_level=True, batch=b, num_patches=p)
    else:
        detector = AnomalyDetector()
        
    # retrieve normality
    if model.memory_bank.shape[0] > 1000:
        print(' retrieve normality, from memory bank')
        normality = model.memory_bank
    else:
        print(' not enough data in memory bank, sampling new data trom train set')
        # patch localization break down images to patches
        # every patch used for normality data
        # 1 or 2 images needed (800-1600 about patches for normality)
        if mvtec_inference:
            normality_datamodule = MVTecDatamodule(
                dataset_dir,
                batch_size=1,
            )
        else:
            normality_datamodule = PretextTaskDatamodule(
                subject=subject,
                root_dir=dataset_dir,
                batch_size=1,
            )
        normality_datamodule.setup()
        normality_retriever = pl.Trainer(accelerator='auto', devices=1, limit_predict_batches=2 if patch_localization else -1)
        output_normality:ModelOutputsContainer = normality_retriever.predict(model, dataloaders=normality_datamodule.train_dataloader())[0]
        output_normality.to_cpu()
        normality = output_normality.embedding_vectors
        
    output.to_cpu()
    
    detector.fit(normality)
    print(' computing anomaly scores')
    anomaly_scores = detector.predict(output.embedding_vectors)
    
    output.threshold = detector.threshold
    output.anomaly_maps = anomaly_scores
    return output


# function for upsampling anomaly maps
def upsample(anomaly_maps:Tensor, target_size:int=256) -> Tensor:
    #print('>>> upsampling')
    ksize = 7
    anomaly_maps = F.relu(functional.gaussian_blur(anomaly_maps, kernel_size=ksize))
    anomaly_maps = F.interpolate(anomaly_maps, target_size, mode='bilinear')
    return anomaly_maps # return tensor shape is Bx1xHxW
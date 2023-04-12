from PIL import Image
from tqdm import tqdm
from self_supervised.gradcam import GradCam
from torchvision import transforms
from torchvision.transforms import functional
from self_supervised.functional import *
from self_supervised.converters import *
from self_supervised.visualization import *
from torch.nn import functional as F
import self_supervised.datasets as dt
import self_supervised.models as md
import random
import os
import torch
import numpy as np



class Localizer:
    def __init__(
            self,
            dataset_dir:str=None,
            root_input_dir:str=None, 
            root_output_dir:str=None,
            subject:str=None,
            model_name:str='best_model.ckpt',
            patch_localization=True,
            imsize:tuple=(256,256),
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
        self.imsize = imsize
        self.patch_localization = patch_localization
        self.patch_dim = patch_dim
        self.stride = stride
        self.threshold = threshold
        
        random.seed(seed)
        np.random.seed(seed)
        
        if dataset_dir:
            self.setup_dataset(imsize=imsize)
    
    
    def _get_detector_good_embeddings(self, img:Tensor=None):
        model:md.PeraNet = md.PeraNet.load_from_checkpoint(self.model_dir+self.model_name)
        model.enable_mvtec_inference()
        model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')
        sample_imgs = 3
        if sample_imgs > len(self.mvtec.train_dataloader().dataset.images_filenames):
            sample_imgs = len(self.mvtec.train_dataloader().dataset.images_filenames)
        tot_embeddings = []
        pbar = tqdm(range(sample_imgs), desc='train data')
        choices = [i for i in range(0, len(self.mvtec.train_dataloader().dataset.images_filenames))]
        idxs = random.choices(choices, k=sample_imgs)
        for i in pbar:
            idx = idxs[i]
            x, _, _ = self.mvtec.train_dataloader().dataset.__getitem__(idx)
            x_patches = extract_patches(x.unsqueeze(0), self.patch_dim, 8)
            if torch.cuda.is_available():
                with torch.no_grad():
                    output = self.model(x_patches.to('cuda'))
            else:
                with torch.no_grad():
                    output = self.model(x_patches)
            patches_embeddings = output['latent_space'].to('cpu')
            y_hats = get_prediction_class(output['classifier']).to('cpu')
            for j in range(len(patches_embeddings)):
                #if y_hats[i] == 0:
                #    tot_embeddings.append(np.array(patches_embeddings[i]))
                tot_embeddings.append(np.array(patches_embeddings[j]))
            #tot_embeddings.append(patches_embeddings.to('cpu')[None, :])
        train_tot_embeddings = torch.tensor(np.array(tot_embeddings))
        #a,b,c = tot_embeddings.shape
        #tot_embeddings_reshaped = torch.reshape(tot_embeddings, (a*b, c))

        
        return train_tot_embeddings
    
    
    def setup_model(self):
        print('setting up models')
        self.model:md.PeraNet = md.PeraNet.load_from_checkpoint(self.model_dir+self.model_name)
        self.model.enable_mvtec_inference()
        self.model.eval()
        if not self.patch_localization:
            print('preparing gradcam')
            self.gradcam = GradCam(
                md.PeraNet.load_from_checkpoint(self.model_dir+self.model_name))
        else:
            if torch.cuda.is_available():
                self.model.to('cuda')
            self.detector = md.AnomalyDetector()
            #if self.model.memory_bank.numel() > 0:
            #    self.detector.fit(self.model.memory_bank)
            #else:
            #    self.detector.fit(self._get_detector_good_embeddings())
            x1 = self._get_detector_good_embeddings()
            self.detector.fit(x1)
            
        
    def setup_dataset(self, imsize:tuple=(256,256)):
        
        self.mvtec = dt.MVTecDatamodule(
            root_dir=self.dataset_dir+self.subject+'/',
            imsize=imsize,
            batch_size=64,
            subject=self.subject
        )
        self.mvtec.setup()
    
    
    def localize(self, num_images:int=10):
        print('starting localization')
        j = len(self.mvtec.test_dataset)
        choices = [i for i in range(0,j)]
        images_idx = random.choices(choices, k=num_images)
        for i in tqdm(range(num_images), desc='test images'):
            idx = images_idx[i]
            x_prime, gt, x = self.mvtec.test_dataset[idx]
            if not self.patch_localization:
                with torch.no_grad():
                    predictions = self.model(x_prime[None, :])
                y_hat = get_prediction_class(predictions['classifier'].to('cpu'))
                if y_hat == 0:
                    saliency_map = torch.zeros(self.imsize)[None, :]
                else:
                    saliency_map = self.gradcam(x_prime[None, :], y_hat)
            else: 
                patches = extract_patches(x_prime[None, :], self.patch_dim, self.stride)
                if torch.cuda.is_available():
                    patches = patches.to('cuda')
                with torch.no_grad():
                    outputs = self.model(patches)
                embeddings = outputs['latent_space'].to('cpu')
                anomaly_scores = self.detector.predict(embeddings)
                dim = int(np.sqrt(embeddings.shape[0]))
                #saliency_map = normalize(anomaly_scores)
                saliency_map = torch.reshape(anomaly_scores, (dim, dim))
                plot_single_image(saliency_map, self.outputs_dir+'/'+str(i)+'_img/', name=str(i)+'_anomaly_map_smol.png')
                #saliency_map = saliency_map*0.8
                ksize = 3
                saliency_map = functional.gaussian_blur(saliency_map[None,:], kernel_size=ksize).squeeze()
                saliency_map = F.relu(saliency_map)
                saliency_map = F.interpolate(saliency_map[None,None,:], self.imsize[0], mode='bilinear').squeeze()
                saliency_map[saliency_map < 0.] = 0.
                saliency_map[saliency_map > 1.] = 1.
            print(torch.max(saliency_map))
            print(torch.min(saliency_map))
            
            #plot_saliency(saliency_map, self.outputs_dir+'saliency_maps/', 'saliency_'+str(i)+'.png')
            image = imagetensor2array(x)
            gt = imagetensor2array(gt)
            heatmap = apply_heatmap(x[None, :], saliency_map[None, :])
            predicted_mask = saliency_map > self.detector.threshold
            segmentation = apply_segmentation(image, np.array(predicted_mask))
            plot_single_image(image, self.outputs_dir+'/'+str(i)+'_img/', name=str(i)+'_original.png') 
            plot_single_image(saliency_map, self.outputs_dir+'/'+str(i)+'_img/', name=str(i)+'_anomaly_map.png')
            plot_single_image(heatmap, self.outputs_dir+'/'+str(i)+'_img/', name=str(i)+'_heatmap.png')   
            plot_single_image(gt, self.outputs_dir+'/'+str(i)+'_img/', name=str(i)+'_ground_truth.png')
            plot_single_image(predicted_mask, self.outputs_dir+'/'+str(i)+'_img/', name=str(i)+'_predicted_mask.png')
            plot_single_image(segmentation, self.outputs_dir+'/'+str(i)+'_img/', name=str(i)+'_segmentation.png')
            #plot_original_and_saliency(image, saliency_map)
            #plot_original_saliency_segmentation(image, saliency_map, segmentation)
            #gt = imagetensor2array(gt)
            #saliency_map = (255*np.array(saliency_map)).astype(np.uint8)
            plot_heatmap_and_masks(
                image=image,
                heatmap=saliency_map,
                gt_mask=gt,
                predicted_mask=segmentation,
                saving_path=self.outputs_dir+'/'+str(i)+'_img/',
                name=str(i)+'_final_result'+'.png'
            )


    def localize_single_image(
            self, 
            filename:str, 
            imsize:tuple=(256,256), 
            output_name:str='output.png'):
        
        image = Image.open(filename).resize(imsize).convert('RGB')
        input_tensor = transforms.ToTensor()(image)
        input_tensor_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(input_tensor)
        if not self.patch_localization:
            predictions = self.model(input_tensor_norm[None, :])
            y_hat = predictions['classifier']
            if y_hat[0] == 0:
                saliency_map = torch.zeros((256,256))[None, :]
            else:
                saliency_map = self.gradcam(input_tensor_norm[None, :], y_hat[0])
        
        heatmap = apply_heatmap(input_tensor[None, :], saliency_map)
        image_array = imagetensor2array(input_tensor)
        plot_heatmap(image_array, heatmap, saving_path=self.outputs_dir, name=output_name)


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


if __name__ == "__main__":
    dataset_dir='dataset/'
    root_inputs_dir='brutta_copia/outputs/patch_level/computations/'
    root_outputs_dir='brutta_copia/b/patch_level/localization/'
    imsize=(256,256)
    patch_dim = 32
    stride=8
    seed=248826279
    patch_localization=True
      
    experiments = get_all_subject_experiments('dataset/')
    textures = get_textures_names()
    obj1 = obj_set_one()
    obj2 = obj_set_two()
    
    experiments_list = ['carpet']
    pbar = tqdm(range(len(experiments_list)), position=0, leave=False)
    for i in pbar:
        pbar.set_description('Localization pipeline | current subject is '+experiments_list[i].upper())
        localizer = Localizer(
            dataset_dir=dataset_dir,
            root_input_dir=root_inputs_dir,
            root_output_dir=root_outputs_dir,
            subject=experiments_list[i],
            model_name='best_model.ckpt',
            imsize=imsize,
            patch_localization=patch_localization,
            patch_dim=patch_dim,
            stride=stride,
            seed=seed
        )
        localizer.setup_model()
        localizer.localize(num_images=3)
        #os.system('clear')
        
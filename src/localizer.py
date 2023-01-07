from PIL import Image
from tqdm import tqdm
from self_supervised.gradcam import GradCam
from torchvision import transforms
from self_supervised.support.functional import *
from self_supervised.support.visualization import *
import self_supervised.support.constants as CONST
import self_supervised.datasets as dt
import self_supervised.model as md
import self_supervised.metrics as mtr
import random
import os
import torch
import numpy as np
import pytorch_lightning as pl



class Localizer:
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
        
        if dataset_dir:
            self.setup_dataset()
    
    
    def _get_kde_embeddings(self):
        model:md.PeraNet = md.PeraNet.load_from_checkpoint(self.model_dir+self.model_name)
        model.enable_mvtec_inference()
        model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')
        sample_imgs = 1
        tot_embeddings = []
        for i in range(sample_imgs):
            x, _, _ = self.mvtec.train_dataloader().dataset.__getitem__(i)
            x_patches = extract_patches(x.unsqueeze(0), self.patch_dim, self.stride)
            if torch.cuda.is_available():
                with torch.no_grad():
                    output = self.model(x_patches.to('cuda'))
            else:
                with torch.no_grad():
                    output = self.model(x_patches)
            patches_embeddings = output['latent_space']
            tot_embeddings.append(patches_embeddings.to('cpu')[None, :])
        tot_embeddings = torch.cat(tot_embeddings, dim=0)
        a,b,c = tot_embeddings.shape
        tot_embeddings = torch.reshape(tot_embeddings, (a*b, c))
        return tot_embeddings
    
    
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
            print('preparing kde train data')
            self.kde = md.MahalanobisDistance()
            self.kde.fit(self._get_kde_embeddings())
            
        
    def setup_dataset(self, imsize:tuple=(256,256)):
        
        self.mvtec = dt.MVTecDatamodule(
            root_dir=self.dataset_dir+self.subject+'/',
            subject=self.subject,
            imsize=imsize,
            batch_size=64
        )
        self.mvtec.setup()
    
    
    def localize(self, num_images:int=10):
        print('starting localization')
        j = len(self.mvtec.test_dataset)-1
        for i in tqdm(range(num_images), desc='images'):
            x_prime, gt, x = self.mvtec.test_dataset[random.randint(0, j)]
            if not self.patch_localization:
                with torch.no_grad():
                    predictions = self.model(x_prime[None, :])
                y_hat = get_prediction_class(predictions['classifier'].to('cpu'))
                if y_hat == 0:
                    saliency_map = torch.zeros((256,256))[None, :]
                else:
                    saliency_map = self.gradcam(x_prime[None, :], y_hat)
            else: 
                patches = extract_patches(x_prime[None, :], self.patch_dim, self.stride)
                if torch.cuda.is_available():
                    patches = patches.to('cuda')
                with torch.no_grad():
                    outputs = self.model(patches)
                embeddings = outputs['latent_space'].to('cpu')
                anomaly_scores = self.kde.predict(embeddings)
                anomaly_scores = normalize(anomaly_scores)
                dim = int(np.sqrt(embeddings.shape[0]))
                saliency_map = torch.reshape(anomaly_scores, (dim, dim))
                saliency_map = F.interpolate(saliency_map[None,None,:], 256, mode='bilinear')
            heatmap = apply_heatmap(x[None, :], saliency_map)
            image = imagetensor2array(x)
            gt = imagetensor2array(gt)
            plot_heatmap_and_masks(
                image=image,
                heatmap=heatmap,
                gt_mask=gt,
                saving_path=self.outputs_dir,
                name='heatmap_and_masks_'+str(i)+'.png'
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
    root_inputs_dir='outputs/computations/'
    root_outputs_dir='outputs/localization/'
    num_images=3,
    imsize=(256,256),
    seed=204110176,
    patch_localization=False
      
    experiments = get_all_subject_experiments('dataset/')
    textures = get_textures_names()
    obj1 = obj_set_one()
    obj2 = obj_set_two()
    
    experiments_list = experiments
    pbar = tqdm(range(len(experiments_list)), position=0, leave=False)
    for i in pbar:
        pbar.set_description('Localization pipeline | current subject is '+experiments_list[i].upper())
        localizer = Localizer(
            dataset_dir=dataset_dir,
            root_input_dir=root_inputs_dir,
            root_output_dir=root_outputs_dir,
            subject=experiments_list[i],
            model_name='best_model.ckpt',
            patch_localization=False,
            seed=0
        )
        localizer.setup_model()
        localizer.localize()
        os.system('clear')
        
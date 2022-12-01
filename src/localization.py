from PIL import Image
from tqdm import tqdm
from self_supervised.gradcam import GradCam
from torchvision import transforms
from self_supervised.support.functional import *
from self_supervised.support.visualization import localize, plot_heatmap, plot_heatmap_and_masks
import self_supervised.support.constants as CONST
import self_supervised.datasets as dt
import self_supervised.model as md
import random
import os
import torch
import numpy as np



def localization_pipeline(
        dataset_dir:str, 
        root_dir:str, 
        subject:str,
        level:str,
        seed=CONST.DEFAULT_SEED()):
    
    imsize = CONST.DEFAULT_IMSIZE()
    batch_size = CONST.DEFAULT_BATCH_SIZE()
    gradcam_dir = root_dir+subject+'/'+level+'/gradcam/'
    model_dir = root_dir+subject+'/'+level+'/'+CONST.DEFAULT_CHECKPOINT_MODEL_NAME()
    
    print('>>> preparing datamodule')
    mvtec = dt.MVTecDatamodule(
        root_dir=dataset_dir+subject+'/',
        subject=subject,
        imsize=imsize,
        batch_size=batch_size,
        seed=seed,
        localization=True
    )
    mvtec.setup()
    sslm = md.SSLM.load_from_checkpoint(
    model_dir)
    sslm.eval()
    
    gradcam = GradCam(
        md.SSLM.load_from_checkpoint(model_dir).model)
    j = len(mvtec.test_dataset)-1
    random.seed(seed)
    for i in tqdm(range(5), desc='localizing defects'):
        input_image_tensor, gt = mvtec.test_dataset[random.randint(0, j)]
        input_tensor_norm = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(input_image_tensor)
        
        prediction_raw, _ = sslm(input_tensor_norm[None, :])
        predicted_class = get_prediction_class(prediction_raw)
        if predicted_class == 0:
            saliency_map = torch.zeros(imsize)[None, None, :]
        else:
            if predicted_class > 1:
                predicted_class = 1
            saliency_map = gradcam(input_tensor_norm[None, :], predicted_class)
        heatmap = localize(input_image_tensor[None, :], saliency_map)
        image = imagetensor2array(input_image_tensor)
        gt = imagetensor2array(gt)
        plot_heatmap(image, heatmap, saving_path=gradcam_dir)
        
        plot_heatmap_and_masks(
            image, 
            heatmap, 
            gt, 
            heatmap2mask(saliency_map.squeeze(), threshold=0.75),
            saving_path=gradcam_dir)
        os.system('clear')
      

def localize_single_image(
        image_filename:str, 
        results_dir:str,
        subject:str):
    
    gradcam_dir = 'memes/results/'
    imsize = CONST.DEFAULT_IMSIZE()
    
    #image
    input_image = Image.open(image_filename).resize(imsize).convert('RGB')
    input_image_tensor = transforms.ToTensor()(input_image)
    input_tensor_norm = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(input_image_tensor)
    #model
    sslm = md.SSLM.load_from_checkpoint(
        results_dir+subject+'/'+CONST.DEFAULT_CHECKPOINT_MODEL_NAME())
    sslm.eval()
    #gradcam wrapper
    gradcam = GradCam(
        md.SSLM.load_from_checkpoint(
            results_dir+subject+'/'+CONST.DEFAULT_CHECKPOINT_MODEL_NAME()).model)
    #predictions
    prediction_raw, _ = sslm(input_tensor_norm[None, :])
    predicted_class = get_prediction_class(prediction_raw)
    if predicted_class == 0:
        title = 'good'
        saliency_map = torch.zeros(imsize)[None, None, :]
    else:
        title = 'defect'
        if predicted_class > 1:
            predicted_class = 1
        saliency_map = gradcam(input_tensor_norm[None, :], predicted_class)
    #localization
    heatmap = localize(input_image_tensor[None, :], saliency_map)
    image = imagetensor2array(input_image_tensor)
    #gt = imagetensor2array(gt)
    plot_heatmap(image, heatmap, results_dir=gradcam_dir, name='broken_large', title=title)


def pipeline():
    dataset_dir = 'dataset/'
    root_dir = 'outputs/computations/'
    seed = 0
    localization_pipeline(
        dataset_dir=dataset_dir,
        root_dir=root_dir,
        subject='grid',
        level='image_level',
        seed=seed
    )


def singleim():
    results_dir = 'temp/computations/'
    localize_single_image(
        image_filename='dataset/bottle/test/broken_large/000.png',
        results_dir=results_dir,
        subject='bottle'
    )


if __name__ == "__main__":
    pipeline()
    #singleim()
    
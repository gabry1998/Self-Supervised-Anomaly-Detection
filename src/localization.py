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



def image_localization(mvtec, model_dir, gradcam_dir, seed):
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
            saliency_map = torch.zeros((256,256))[None, None, :]
        else:
            if predicted_class > 1:
                predicted_class = 1
            saliency_map = gradcam(input_tensor_norm[None, :], predicted_class)
        heatmap = localize(input_image_tensor[None, :], saliency_map)
        image = imagetensor2array(input_image_tensor)
        gt = imagetensor2array(gt)
        plot_heatmap(image, heatmap, saving_path=gradcam_dir, name='heatmap_'+str(i)+'.png')
        
        plot_heatmap_and_masks(
            image, 
            heatmap, 
            gt, 
            heatmap2mask(saliency_map.squeeze(), threshold=0.75),
            saving_path=gradcam_dir,
            name='heatmap_and_masks_'+str(i)+'.png')
        #os.system('clear')


def patch_localization(mvtec, model_dir, gradcam_dir, seed):
    sslm = md.SSLM.load_from_checkpoint(
    model_dir)
    sslm.to('cuda')
    sslm.eval()
    sslm.unfreeze_layers(False)
    
    train_embeddings_gde = []
    for i in range(5):
        train_img_tensor, _ = mvtec.train_dataloader().dataset.__getitem__(i)
        train_img_tensor_norm = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(train_img_tensor)
        train_img_tensor_norm = train_img_tensor_norm.unsqueeze(0)
        
        train_patches = extract_patches(train_img_tensor_norm, 32, 4)
        _, train_embedding = sslm(train_patches.to('cuda'))
        train_embeddings_gde.append(train_embedding.to('cpu'))
    train_embedding = torch.cat(train_embeddings_gde, dim=0)
    gde = md.GDE()
    gde.fit(train_embedding)
    j = len(mvtec.test_dataset)-1
    random.seed(seed)
    for i in tqdm(range(5), desc='localizing defects'):
        input_image_tensor, gt = mvtec.test_dataset[random.randint(0, j)]
        input_tensor_norm = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(input_image_tensor)
        input_tensor_norm = input_tensor_norm.unsqueeze(0)
        patches = extract_patches(input_tensor_norm, 32, 4)
        y_hat, embeddings = sslm(patches.to('cuda'))
        y_hat = get_prediction_class(y_hat.to('cpu'))
        embeddings = embeddings.to('cpu')
        mvtec_test_scores = gde.predict(embeddings)
        dim = int(np.sqrt(embeddings.shape[0]))
        out = torch.reshape(mvtec_test_scores, (dim, dim))
        out = normalize(out)
        out[out < 0.35] = 0
        gs = GaussianSmooth(device='cpu')
        out = gs.upsample(out[None, None, :])
        out = normalize(out)
        
        heatmap = localize(input_image_tensor[None, :], out)
        image = imagetensor2array(input_image_tensor)
        heatmap = np.uint8(255 * heatmap)
        image = np.uint8(255 * image)
        plot_heatmap(image, heatmap, saving_path=gradcam_dir, name='heatmap_'+str(i)+'.png')
        gt = imagetensor2array(gt)
        plot_heatmap_and_masks(
            image, 
            heatmap, 
            gt, 
            heatmap2mask(out.squeeze(), threshold=0.7),
            saving_path=gradcam_dir,
            name='heatmap_and_masks_'+str(i)+'.png')
        #os.system('clear')
    
    
def localization_pipeline(
        dataset_dir:str,
        root_inputs_dir:str,
        root_outputs_dir:str,
        subject:str,
        level:str,
        seed=CONST.DEFAULT_SEED()):
    
    imsize = CONST.DEFAULT_IMSIZE()
    batch_size = CONST.DEFAULT_BATCH_SIZE()
    gradcam_dir = root_outputs_dir+subject+'/'+level+'/gradcam/'
    model_dir = root_inputs_dir+subject+'/'+level+'/'+'best_model.ckpt'
    
    mvtec = dt.MVTecDatamodule(
        root_dir=dataset_dir+subject+'/',
        subject=subject,
        imsize=imsize,
        batch_size=batch_size,
        seed=seed,
        localization=True
    )
    mvtec.setup()
    
    if level=='image_level':
        image_localization(mvtec, model_dir, gradcam_dir, seed)
    if level=='patch_level':
        patch_localization(mvtec, model_dir, gradcam_dir, seed)

def pipeline():
    root_outputs_dir='brutta_copia/bho/'
    experiments = np.array([
        'bottle',
        'grid',
        'tile',
        'toothbrush',
        'screw'
    ])
    level = 'patch_level'
    pbar = tqdm(range(len(experiments)))
    for i in pbar:
        pbar.set_description('Pipeline Execution '+level+' | current subject is '+experiments[i].upper())
        localization_pipeline(
            dataset_dir='dataset/', 
            root_inputs_dir='outputs/computations/',
            root_outputs_dir=root_outputs_dir,
            subject=experiments[i],
            level=level,
            seed=1
        )
        os.system('clear')

if __name__ == "__main__":
    pipeline()
    #singleim()
    
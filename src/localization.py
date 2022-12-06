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


def localize_single_image(
        filename:str,
        imsize:tuple,
        model_dir:str,
        output_dir:str,
        output_name:str
        ):
    image = Image.open(filename).resize(imsize).convert('RGB')
    input_tensor = transforms.ToTensor()(image)
    input_tensor_norm = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(input_tensor)
    
    sslm = md.SSLM.load_from_checkpoint(
    model_dir)
    sslm.eval()
    sslm.unfreeze_layers(False)

    gradcam = GradCam(
        md.SSLM.load_from_checkpoint(model_dir).model)
    
    y_hat, embedding = sslm(input_tensor_norm[None, :])
    predicted_class = get_prediction_class(y_hat)
    if predicted_class == 0:
        saliency_map = torch.zeros((256,256))[None, None, :]
    else:
        if predicted_class > 1:
            predicted_class = 1
        saliency_map = gradcam(input_tensor_norm[None, :], predicted_class)
    heatmap = localize(input_tensor[None, :], saliency_map)
    image_array = imagetensor2array(input_tensor)
    plot_heatmap(image_array, heatmap, saving_path=output_dir, name=output_name)
    
    
def image_level_localization(
        datamodule:dt.MVTecDatamodule, 
        root_inputs_dir:str,
        root_outputs_dir:str,
        subject:str,
        num_images:int=3):
    
    outputs_dir = root_outputs_dir+subject+'/image_level/gradcam/'
    model_input_dir = root_inputs_dir+subject+'/image_level/best_model.ckpt'
    
    sslm = md.SSLM.load_from_checkpoint(
    model_input_dir)
    sslm.eval()
    gradcam = GradCam(
        md.SSLM.load_from_checkpoint(model_input_dir).model)
    j = len(datamodule.test_dataset)-1
    
    ground_truth_maps = []
    anomaly_maps = []
    
    for i in tqdm(range(num_images), desc='localizing defects'):
        input_image_tensor, gt = datamodule.test_dataset[random.randint(0, j)]
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
        gt_mask = imagetensor2array(gt)
        pred_mask = heatmap2mask(saliency_map.squeeze(), threshold=0.75)
        #plot_heatmap(image, heatmap, saving_path=gradcam_dir, name='heatmap_'+str(i)+'.png')
        
        plot_heatmap_and_masks(
            image, 
            heatmap, 
            gt_mask, 
            pred_mask,
            saving_path=outputs_dir,
            name='heatmap_and_masks_'+str(i)+'.png')
        anomaly_maps.append(np.array(saliency_map.squeeze()))
        ground_truth_maps.append(np.array(gt.squeeze()))
        
    print('>>> PRO and AUPRO')
    all_fprs, all_pros = mtr.compute_pro(
    anomaly_maps=np.array(anomaly_maps),
    ground_truth_maps=np.array(ground_truth_maps))

    au_pro = mtr.compute_aupro(all_fprs, all_pros, 0.3)
    plot_curve(
        all_fprs,
        all_pros,
        au_pro,
        saving_path=root_outputs_dir+subject+'/image_level/',
        title='Pro curve for '+subject.upper(),
        name='pro.png'
    )
    return au_pro


def patch_level_localization( 
        datamodule:dt.MVTecDatamodule, 
        root_inputs_dir:str,
        root_outputs_dir:str,
        subject:str,
        num_images:int=5):
    
    outputs_dir = root_outputs_dir+subject+'/patch_level/gradcam/'
    model_input_dir = root_inputs_dir+subject+'/patch_level/best_model.ckpt'
    
    sslm = md.SSLM.load_from_checkpoint(
    model_input_dir)
    sslm.to('cuda')
    sslm.eval()
    sslm.unfreeze_layers(False)
    
    train_embeddings_gde = []
    train_gde_imgs = random.sample(list(datamodule.train_dataloader().dataset.images_filenames), 2)
    for i in range(len(train_gde_imgs)):
        #print(train_gde_imgs[i])
        train_img_tensor = Image.open(train_gde_imgs[i]).resize((256,256)).convert('RGB')
        train_img_tensor = transforms.ToTensor()(train_img_tensor)
        train_img_tensor_norm = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(train_img_tensor)
        train_img_tensor_norm = train_img_tensor_norm.unsqueeze(0)
        
        train_patches = extract_patches(train_img_tensor_norm, 32, 4)
        _, train_embedding = sslm(train_patches.to('cuda'))
        train_embeddings_gde.append(train_embedding.to('cpu'))
    train_embedding = torch.cat(train_embeddings_gde, dim=0)
    gde = md.GDE1()
    print(train_embedding.shape)
    gde.fit(train_embedding)
    j = len(datamodule.test_dataset)-1
    for i in tqdm(range(num_images), desc='localizing defects'):
        input_image_tensor, gt = datamodule.test_dataset[random.randint(0, j)]
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
        #plot_heatmap(image, heatmap, saving_path=gradcam_dir, name='heatmap_'+str(i)+'.png')
        gt = imagetensor2array(gt)
        plot_heatmap_and_masks(
            image, 
            heatmap, 
            gt, 
            heatmap2mask(out.squeeze(), threshold=0.7),
            saving_path=outputs_dir,
            name='heatmap_and_masks_'+str(i)+'.png')
        
    
def localization_pipeline(
        dataset_dir:str,
        root_inputs_dir:str,
        root_outputs_dir:str,
        subject:str,
        num_images:int=3,
        imsize:tuple=CONST.DEFAULT_IMSIZE(),
        seed:int=CONST.DEFAULT_SEED(),
        patch_localization:bool=False):
    
    random.seed(seed)
    np.random.seed(seed)
    mvtec = dt.MVTecDatamodule(
        root_dir=dataset_dir+subject+'/',
        subject=subject,
        imsize=imsize,
        localization=True
    )
    mvtec.setup()
    #gradcam_dir = root_outputs_dir+subject+'/'+level+'/gradcam/'
    #model_dir = root_inputs_dir+subject+'/'+level+'/'+'best_model.ckpt'
    if patch_localization:
        patch_level_localization(
            datamodule=mvtec, 
            root_inputs_dir=root_inputs_dir, 
            root_outputs_dir=root_outputs_dir,
            subject=subject,
            num_images=num_images)
    else:
        image_level_localization(
            datamodule=mvtec, 
            root_inputs_dir=root_inputs_dir, 
            root_outputs_dir=root_outputs_dir,
            subject=subject,
            num_images=num_images)
    

def run(
        experiments_list:list,
        dataset_dir:str,
        root_inputs_dir:str,
        root_outputs_dir:str,
        num_images:int=3,
        imsize:int=CONST.DEFAULT_IMSIZE(),
        seed:int=0,
        patch_localization=False
        ):
    
    os.system('clear')
    
    pbar = tqdm(range(len(experiments_list)))
    for i in pbar:
        pbar.set_description('Localization pipeline | current subject is '+experiments_list[i].upper())
        subject = experiments_list[i]
        localization_pipeline(
            dataset_dir=dataset_dir,
            root_inputs_dir=root_inputs_dir,
            root_outputs_dir=root_outputs_dir,
            subject=subject,
            num_images=num_images,
            imsize=imsize,
            seed=seed,
            patch_localization=patch_localization
        )


def single_im():
    localize_single_image(
        filename='memes/in/artemisia.jpg',
        imsize=(256,256),
        model_dir='outputs/computations/bottle/image_level/best_model.ckpt',
        output_dir='memes/out/',
        output_name='anomalous_artemisia.png'
    )


if __name__ == "__main__":
    run(
        experiments_list=get_all_subject_experiments('dataset/'),
        dataset_dir='dataset/',
        root_inputs_dir='outputs/computations/',
        root_outputs_dir='brutta_copia/',
        num_images=5,
        imsize=(256,256),
        seed=0,
        patch_localization=False
        )
    #single_im()
    
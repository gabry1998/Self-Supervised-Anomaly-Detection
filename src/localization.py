import os
import self_supervised.support.constants as CONST
import self_supervised.datasets as dt
import self_supervised.model as md
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from self_supervised.support.cutpaste_parameters import CPP
from self_supervised.support.dataset_generator import apply_jittering, generate_patch, paste_patch
from self_supervised.support.functional import ModelLocalizerWrapper, SimilarityToConceptTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM
from torchvision import transforms
import random

def localization_pipeline(
        dataset_dir:str, 
        results_dir:str, 
        subject:str):
    
    imsize = CONST.DEFAULT_IMSIZE()
    batch_size = CONST.DEFAULT_BATCH_SIZE()
    seed = CONST.DEFAULT_SEED()
    
    print('>>> preparing datamodule')
    datamodule = dt.MVTecDatamodule(
        root_dir=dataset_dir+subject+'/',
        subject=subject,
        imsize=imsize,
        batch_size=batch_size,
        seed=seed,
        localization=True
    )
    datamodule.setup()
    x, y = next(iter(datamodule.train_dataloader()))
    
    print('>>> preparing model')
    sslm = md.SSLM.load_from_checkpoint(
        results_dir+subject+'/'+CONST.DEFAULT_CHECKPOINT_MODEL_NAME())
    sslm.eval()
    sslm.set_for_localization(True)
    localizer = ModelLocalizerWrapper(sslm.model)
    
    clone = md.SSLM.load_from_checkpoint(results_dir+subject+'/'+CONST.DEFAULT_CHECKPOINT_MODEL_NAME())
    clone.eval()
    
    references = []
    defect_type = 'good'
    good_reference = Image.open('dataset/'+subject+'/train/'+defect_type+'/000.png').resize(imsize).convert('RGB')
    #patch, coords = generate_patch(good_reference)
    #patch = apply_jittering(patch, CPP.jitter_transforms)
    #defect_reference = paste_patch(good_reference, patch, coords)
    defect_im = transforms.ToTensor()(good_reference)
    defect_im = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(defect_im)
    references.append(defect_im[None, :])
    
    references = torch.cat(references)
    localizer.to('cpu')
    concept_features = localizer(references)
    my_target_layers = [sslm.model.feature_extractor.layer4[-1]]
    defect_reference_target = [SimilarityToConceptTarget(concept_features[0])]

    j = len(datamodule.test_dataset)-1
    for i in tqdm(range(5), desc='Getting image reference features'):
        query, _ = datamodule.test_dataset[random.randint(0, j)]
        x_query = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(query)
        
        y_hat, _ = clone(x_query[None, :])
        y_hat = torch.max(y_hat.data, 1)
        y_hat = int(y_hat.indices)
        my_grayscale_cam = None
        with GradCAM(model=localizer,
                    target_layers=my_target_layers,
                    use_cuda=False) as cam1:
            if y_hat == 0:
                my_grayscale_cam = torch.zeros((256,256))
            else:
                grayscale_cam = cam1(input_tensor=x_query[None, :],
                                    targets=defect_reference_target)
                my_grayscale_cam = grayscale_cam[0,:]
            
        
        my_image_float = np.array(torch.permute(query, (2,1,0)))
        cam_image = show_cam_on_image(my_image_float, np.array(my_grayscale_cam), use_rgb=True)
        
        images = [my_image_float, np.float32(cam_image) / 255]
        output = np.hstack(images)
        
        plt.title(str(y_hat))
        plt.imshow(output)
        if not os.path.exists(results_dir+subject+'/gradcam/'):
            os.makedirs(results_dir+subject+'/gradcam/')
        plt.savefig(results_dir+subject+'/gradcam/'+str(i)+'.png')
        plt.close()

        
        os.system('clear')
        
if __name__ == "__main__":
    dataset_dir = 'dataset/'
    results_dir = 'outputs/computations/'
    localization_pipeline(
        dataset_dir=dataset_dir,
        results_dir=results_dir,
        subject='bottle'
    )
    
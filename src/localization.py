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
from self_supervised.support.functional import ModelLocalizerWrapper, SimilarityToConceptTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM
from torchvision import transforms
import random

def localization_pipeline(
        dataset_dir:str, 
        results_dir:str, 
        subject:str,
        dataset_type_generation:str=CONST.DEFAULT_DATASET_GENERATION(),
        classification_task:str=CONST.DEFAULT_CLASSIFICATION_TASK()):
    
    imsize = CONST.DEFAULT_IMSIZE()
    batch_size = CONST.DEFAULT_BATCH_SIZE()
    seed = CONST.DEFAULT_SEED()
    
    print('>>> preparing datamodule')
    datamodule = dt.MVTecDatamodule(
        root_dir=dataset_dir+subject+'/',
        subject=subject,
        imsize=imsize,
        batch_size=batch_size,
        seed=seed
    )
    datamodule.setup()
    x, y = next(iter(datamodule.train_dataloader()))
    
    print('>>> preparing model')
    sslm = md.SSLM.load_from_checkpoint(
        results_dir+'computations/'+subject+'/'+dataset_type_generation+\
        '/'+classification_task+'/'+CONST.DEFAULT_CHECKPOINT_MODEL_NAME())
    sslm.eval()
    sslm.set_for_localization(True)
    localizer = ModelLocalizerWrapper(sslm.model)
    
    print('>>> getting image reference features')
    #reference = x[0]
    #x_ref = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(reference)
    
    defect_type = 'good'
    reference = Image.open('dataset/'+subject+'/test/'+defect_type+'/000.png').resize(imsize).convert('RGB')
    x_ref = transforms.ToTensor()(reference)
    x_ref = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(x_ref)
    
    localizer.to('cpu')
    
    concept_features = localizer(x_ref[None, :])[0, :]
    
    #x1, y1 = next(iter(datamodule.test_dataloader()))
    j = len(datamodule.test_dataset)
    
    for i in tqdm(range(20)):
        
        query, _ = datamodule.test_dataset[random.randint(0, j)]
        x_query = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(query)
        
        my_target_layers = [sslm.model.feature_extractor.layer4[-1]]
        my_targets = [SimilarityToConceptTarget(concept_features)]
        with GradCAM(model=localizer,
                    target_layers=my_target_layers,
                    use_cuda=False) as cam1:

            grayscale_cam = cam1(input_tensor=x_query[None, :],
                                targets=my_targets)
            
        my_grayscale_cam = grayscale_cam[0,:]

        my_image_float = np.array(torch.permute(query, (2,1,0)))
        #my_image_float = np.float32(my_image_float) / 255
        cam_image = show_cam_on_image(my_image_float, my_grayscale_cam, use_rgb=True)
        plt.imshow(cam_image)
        plt.savefig(results_dir+'localization/'+subject+'/gradcam/'+str(i)+'.png')
        plt.close()

        
        os.system('clear')
        
if __name__ == "__main__":
    dataset_dir = 'dataset/'
    results_dir = 'outputs/'
    localization_pipeline(
        dataset_dir=dataset_dir,
        results_dir=results_dir,
        subject='bottle',
        dataset_type_generation='generative_dataset',
        classification_task='3-way'
    )
    
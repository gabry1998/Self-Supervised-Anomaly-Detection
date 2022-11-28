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
from self_supervised.support.functional import ScoreCAM

from torchvision import transforms
import random

from self_supervised.support.visualization import convert_for_localization, localize

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
    clone = md.SSLM.load_from_checkpoint(results_dir+subject+'/'+CONST.DEFAULT_CHECKPOINT_MODEL_NAME())
    clone.eval()
    resnet_model_dict = dict(type='resnet18', arch=sslm.model.feature_extractor, layer_name='layer4',input_size=imsize)
    resnet_scorecam = ScoreCAM(resnet_model_dict)
    
    j = len(datamodule.test_dataset)-1
    random.seed(0)
    for i in tqdm(range(5), desc='localizing defects'):
        query, _ = datamodule.test_dataset[random.randint(0, j)]
        x_query = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(query)
        
        y_hat, _ = clone(x_query[None, :])
        y_hat = torch.max(y_hat.data, 1)
        y_hat = int(y_hat.indices)
        
        if y_hat == 0:
            title = 'good'
        else:
            title = 'defect'
        if y_hat == 0:
            heatmap = torch.tensor(np.zeros(imsize))
        else:
            scorecam_map = resnet_scorecam(x_query[None, :].to('cuda'))
            scorecam_map = scorecam_map[0]
            heatmap = scorecam_map[0].to('cpu')
        image = torch.permute(query, (2,1,0))
        
        heatmap = convert_for_localization(heatmap)
        image = convert_for_localization(image)
        superimposed = localize(image, heatmap)
        
        hseparator = np.array(Image.new(mode='RGB', size=(6,256), color=(255,255,255))).astype('uint8')
        output = np.hstack([image, hseparator, superimposed])
        
            
        plt.title(title)
        plt.imshow(output)
        plt.axis('off')
        if not os.path.exists(results_dir+subject+'/gradcam/'):
            os.makedirs(results_dir+subject+'/gradcam/')

        plt.savefig(results_dir+subject+'/gradcam/'+str(i)+'.png', bbox_inches='tight')
        plt.close()

        
        os.system('clear')
        
if __name__ == "__main__":
    dataset_dir = 'dataset/'
    results_dir = 'outputs/computations/'
    localization_pipeline(
        dataset_dir=dataset_dir,
        results_dir=results_dir,
        subject='toothbrush'
    )
    
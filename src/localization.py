import os
import self_supervised.support.constants as CONST
import self_supervised.datasets as dt
import self_supervised.model as md
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

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
    if torch.cuda.is_available():
        x = x.to('cuda')
    
    print('>>> preparing model')
    sslm = md.SSLM.load_from_checkpoint(
        results_dir+subject+'/'+dataset_type_generation+\
        '/'+classification_task+'/'+CONST.DEFAULT_CHECKPOINT_MODEL_NAME())
    if torch.cuda.is_available():
        sslm.to('cuda')
    
    sslm.eval()
    sslm.model.set_for_localization(True)

    for i in tqdm(range(5)):
        print('')
        img = x[i]
        print('>>> computing features')
        avg_pool, layer1, layer2, layer3, layer4 = sslm.model.compute_features(img[None, :])
        print('>>> making predictions')
        predictions, _ = sslm(img[None, :])
        y_hat = torch.argmax(predictions).item()
        print(y_hat)
        predictions[:, y_hat].backward()
        g = sslm.model.gradients
        print(layer4.shape)
        for j in range(512):
            layer4[0, j, :, :] *= g[0][j]
        layer4 = layer4.detach()
        heatmap = torch.mean(layer4, dim=1).squeeze()
        heatmap = np.maximum(heatmap, 0)
        
        
        heatmap = heatmap / torch.max(heatmap)
        heatmap = heatmap.numpy()
        
        if y_hat == 0:
            figtitle = 'good'
        else:  
            figtitle = 'defect'
        
        #print(heatmap.shape)
        plt.title(figtitle)
        plt.imshow(heatmap)
        plt.savefig('outputs/localization/bottle/features/'+str(i)+'.png')

        #gs = GaussianSmooth()
        #heatmap = gs.upsample(np.array(torch.tensor(heatmap)[None, :]))
        #print(heatmap.shape)
        #heatmap = heatmap[0]
        img = np.array(torch.permute(img, (2,1,0)))
        img = np.uint8(255 * img)
        heatmap = cv2.resize(heatmap, CONST.DEFAULT_IMSIZE())
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.3 + img
        superimposed_img = np.uint8(255 * superimposed_img / np.max(superimposed_img))
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        plt.title(figtitle)
        plt.imshow(superimposed_img)
        plt.savefig('outputs/localization/bottle/gradcam/'+str(i)+'.png')
        
        
        os.system('clear')
        
if __name__ == "__main__":
    dataset_dir = 'dataset/'
    results_dir = 'outputs/computations/'
    localization_pipeline(
        dataset_dir=dataset_dir,
        results_dir=results_dir,
        subject='bottle',
        dataset_type_generation='generative_dataset',
        classification_task='3-way'
    )
    
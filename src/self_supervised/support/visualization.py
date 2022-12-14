from sklearn.metrics import roc_curve, auc
from torch import Tensor
from sklearn.manifold import TSNE
from PIL import Image
from self_supervised.support.functional import imagetensor2array, normalize
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import os




def plot_history(network_history, saving_path=None, mode='training'):
    if saving_path and not os.path.exists(saving_path):
        os.makedirs(saving_path)
        
    x_plot = list(range(1,len(network_history['train']['loss'])+1))
    x_plot_val = list(range(1,len(network_history['val']['loss'])+1))
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(x_plot, network_history['train']['loss'])
    plt.plot(x_plot_val, network_history['val']['loss'])
    plt.legend(['Training', 'Validation'])
    if saving_path:
        plt.savefig(saving_path+mode+'_loss.png')
    else:
        plt.savefig(mode+'_loss.png')
    plt.close()

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(x_plot, network_history['train']['accuracy'])
    plt.plot(x_plot_val, network_history['val']['accuracy'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    if saving_path:
        plt.savefig(saving_path+mode+'_accuracy.png')
    else:
        plt.savefig(mode+'_accuracy.png')
    plt.show()
    plt.close()

def plot_curve(fpr, tpr, area, saving_path:str=None, title:str='', name:str='curve.png'):
    if saving_path and not os.path.exists(saving_path):
        os.makedirs(saving_path)
        
    #plot roc
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='Curve (area = %0.2f)' % area)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    
    if saving_path:
        plt.savefig(saving_path+name)
    else:
        plt.savefig(name)
    plt.close()


def plot_tsne(embeddings:Tensor, labels:Tensor, saving_path:str=None, title:str='', name:str='tsne.png', num_classes:int=2):
    if saving_path and not os.path.exists(saving_path):
        os.makedirs(saving_path)
        
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(embeddings.detach().numpy())
    tx = tsne_results[:, 0]
    ty = tsne_results[:, 1]
    df = pd.DataFrame()
    l = labels.tolist()
    l = ['good' if str(x)=='0' else x for x in l]
    l = ['cutpaste' if str(x)=='1' else x for x in l]
    if num_classes == 3:
        l = ['scar' if str(x)=='2' else x for x in l]
        labels = ['mvtec' if str(x)=='3' else x for x in l]
    if num_classes == 2:
        labels = ['mvtec' if str(x)=='2' else x for x in l]
    df["labels"] = labels
    df["comp-1"] = normalize(tx)
    df["comp-2"] = normalize(ty)
    plt.figure()

    sns.scatterplot(hue='labels',
                    x='comp-1',
                    y='comp-2',
                    palette=dict(good='#00B121', cutpaste='#69140E', scar='#A44200', mvtec='#7BB2D9'),
                    data=df).set(title=title)
    if saving_path:
        plt.savefig(saving_path+name)
    else:
        plt.savefig(name)
    plt.close()

def plot_heatmap(image, heatmap, saving_path:str=None, name:str='gradcam.png'):
    if saving_path and not os.path.exists(saving_path):
        os.makedirs(saving_path)
    
    fig, axs = plt.subplots(1,2, figsize=(16,16))
    
    axs[0].axis('off')
    axs[0].set_title('original')
    axs[0].imshow(image)
    
    axs[1].axis('off')
    axs[1].set_title('localization')
    axs[1].imshow(heatmap)
    
    if saving_path:
        plt.savefig(saving_path+name, bbox_inches='tight')
    else:
        plt.savefig(name, bbox_inches='tight')
    plt.close()
    

def plot_heatmap_and_masks(
        image, 
        heatmap, 
        gt_mask, 
        predicted_mask=None, 
        saving_path:str=None, 
        name:str='heatmap_and_masks.png'):
    
    if saving_path and not os.path.exists(saving_path):
        os.makedirs(saving_path)
    if not predicted_mask==None:
        fig, axs = plt.subplots(1,4, figsize=(16,16))
    else:
        fig, axs = plt.subplots(1,3, figsize=(16,16))
    
    axs[0].axis('off')
    axs[0].set_title('original')
    axs[0].imshow(image)
    
    axs[1].axis('off')
    axs[1].set_title('groundtruth')
    axs[1].imshow(np.array(gt_mask, dtype=float))
    
    axs[2].axis('off')
    axs[2].set_title('localization')
    axs[2].imshow(heatmap)
    
    if not predicted_mask==None:
        axs[3].axis('off')
        axs[3].set_title('predicted mask')
        axs[3].imshow(np.array(predicted_mask, dtype=float))
    if saving_path:
        plt.savefig(saving_path+name, bbox_inches='tight')
    else:
        plt.savefig(name, bbox_inches='tight')
    plt.close()

def apply_heatmap(image:Tensor, heatmap:Tensor):
    #image is (1, 3, H, W)
    #heatmap is (1, 1, H, W)
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap.squeeze()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    result = heatmap+image
    result = result.div(result.max()).squeeze()
    return imagetensor2array(result)
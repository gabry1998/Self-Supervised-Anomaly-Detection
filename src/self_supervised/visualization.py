from sklearn.metrics import roc_curve, auc
from torch import Tensor
from sklearn.manifold import TSNE
from skimage import feature
from PIL import Image, ImageFilter
from self_supervised.functional import normalize
from self_supervised.converters import imagetensor2array
from torchvision import transforms
from matplotlib import _color_data as cd
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

def plot_multiple_curve(roc_curves:list, names:list, saving_path:str=None, title:str='', name:str='multi_curve.png'):
    if saving_path and not os.path.exists(saving_path):
        os.makedirs(saving_path)
    
    colors = list(cd.TABLEAU_COLORS)
    n = len(roc_curves)
    plt.figure()
    lw = 2
    for i in range(n):
        item = roc_curves[i]
        fpr = item[0]
        tpr = item[1]
        
        plt.plot(fpr, tpr, color=colors[i],
                lw=lw, label=names[i])
        
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
    
    
def plot_curve(fpr, tpr, area, threshold=None, saving_path:str=None, title:str='', name:str='curve.png'):
    if saving_path and not os.path.exists(saving_path):
        os.makedirs(saving_path)
        
    #plot roc
    plt.figure()
    lw = 2
    if not threshold is None:
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label=str('Curve (area = %0.2f)' % area)+'\n'+str('Optimal threshold: %0.2f' % threshold))
    else:
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label=str('Curve (area = %0.2f)' % area))
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


def plot_tsne(embeddings:Tensor, labels:Tensor, saving_path:str=None, title:str='', name:str='tsne.png'):
    if saving_path and not os.path.exists(saving_path):
        os.makedirs(saving_path)
        
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(embeddings.detach().numpy())
    tx = tsne_results[:, 0]
    ty = tsne_results[:, 1]
    df = pd.DataFrame()
    l = labels.tolist()
    l = ['mvtec_good' if str(x)=='-1' else x for x in l]
    l = ['good' if str(x)=='0' else x for x in l]
    l = ['polygon' if str(x)=='1' else x for x in l]
    l = ['rectangle' if str(x)=='2' else x for x in l]
    l = ['line' if str(x)=='3' else x for x in l]
    labels = ['mvtec_defect' if str(x)=='4' else x for x in l]
    df["labels"] = labels
    df["comp-1"] = normalize(tx)
    df["comp-2"] = normalize(ty)
    plt.figure()

    sns.scatterplot(hue='labels',
                    x='comp-1',
                    y='comp-2',
                    palette=dict(
                        mvtec_good='#59ff00',
                        good='#00B121', 
                        polygon='#69140E', 
                        rectangle='#A44200', 
                        line='orange',
                        mvtec_defect='#7BB2D9'),
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


def apply_segmentation(image:np.ndarray, predicted_mask:np.ndarray):
    edged_image = feature.canny(predicted_mask)
    color_fill = np.array([200, 62, 115], dtype='uint8')
    color_border = np.array([51, 16, 103], dtype='uint8')
    masked_img = np.where(predicted_mask[...,None], color_fill, image)
    out = cv2.addWeighted(image, 0.7, masked_img, 0.3,0)
    masked_img = np.where(edged_image[...,None], color_border, out)
    out = cv2.addWeighted(out, 0.01, masked_img, 0.99,0)
    return out


def plot_single_image(img, saving_path, name):
    plt.set_cmap('magma')
    if saving_path and not os.path.exists(saving_path):
        os.makedirs(saving_path)
    plt.imshow(img, vmin=0, vmax=1)
    plt.axis('off')
    if saving_path:
        plt.savefig(saving_path+name, bbox_inches='tight')
    else:
        plt.savefig(name, bbox_inches='tight')
    plt.close()

def plot_original_and_saliency(image, saliency, saving_path, name):
    if saving_path and not os.path.exists(saving_path):
        os.makedirs(saving_path)
    fig, axs = plt.subplots(1,2, figsize=(16,16))
    axs[0].axis('off')
    axs[0].set_title('original')
    axs[0].imshow(image)
    
    axs[1].axis('off')
    axs[1].set_title('anomaly map')
    axs[1].imshow(saliency)
    if saving_path:
        plt.savefig(saving_path+name, bbox_inches='tight')
    else:
        plt.savefig(name, bbox_inches='tight')
    plt.close()

def plot_original_saliency_segmentation(
        image, 
        saliency, 
        segmentation,
        saving_path:str=None, 
        name:str='segmentation.png'):
    if saving_path and not os.path.exists(saving_path):
        os.makedirs(saving_path)
    fig, axs = plt.subplots(1,3, figsize=(16,16))
    
    axs[0].axis('off')
    axs[0].set_title('original')
    axs[0].imshow(image, vmin=0, vmax=1)
    
    axs[1].axis('off')
    axs[1].set_title('anomaly map')
    axs[1].imshow(saliency, vmin=0, vmax=1)
    
    axs[2].axis('off')
    axs[2].set_title('segmentation')
    axs[2].imshow(segmentation, vmin=0, vmax=1)
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
    if not predicted_mask is None:
        fig, axs = plt.subplots(1,4, figsize=(16,16))
    else:
        fig, axs = plt.subplots(1,3, figsize=(16,16))
    
    axs[0].axis('off')
    axs[0].set_title('original')
    axs[0].imshow(image, vmin=0, vmax=1)
    
    axs[1].axis('off')
    axs[1].set_title('groundtruth')
    axs[1].imshow(gt_mask, vmin=0, vmax=1)
    
    axs[2].axis('off')
    axs[2].set_title('anomaly map')
    axs[2].imshow(heatmap, vmin=0, vmax=1)
    
    if not predicted_mask is None:
        axs[3].axis('off')
        axs[3].set_title('segmentation')
        axs[3].imshow(predicted_mask, vmin=0, vmax=1)
    if saving_path:
        plt.savefig(saving_path+name, bbox_inches='tight')
    else:
        plt.savefig(name, bbox_inches='tight')
    plt.close()


def apply_heatmap(image:Tensor, heatmap:Tensor):
    #image is (1, 3, H, W)
    #heatmap is (1, 1, H, W)
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap.squeeze()), cv2.COLORMAP_MAGMA)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    result = heatmap+image
    result = result.div(result.max()).squeeze()
    return imagetensor2array(result)
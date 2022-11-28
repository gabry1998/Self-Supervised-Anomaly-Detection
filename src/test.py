from self_supervised.datasets import GenerativeDatamodule, MVTecDatamodule
from self_supervised.model import *
from self_supervised.support.dataset_generator import generate_dataset
from self_supervised.support.functional import *
from self_supervised.support.cutpaste_parameters import CPP
from tqdm import tqdm
import time
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
import random
from torchvision import transforms
import glob
import pandas as pd
import math
import self_supervised.model as md
import cv2


def read_test():
    x = get_image_filenames('dataset/bottle/train/good/')
    print(x[0])


def duplicate_test():
    dataset_dir = '/home/ubuntu/TesiAnomalyDetection/dataset/'
    subject = 'toothbrush'

    x = get_image_filenames(dataset_dir+subject+'/train/good/')
    y = get_image_filenames(dataset_dir+subject+'/test/good/')

    print(x.shape)
    print(y.shape)

    x = duplicate_filenames(x)
    y = duplicate_filenames(y)

    print(x.shape)
    print(y.shape)


def gt_test():
    dataset_dir = 'dataset/'
    subject = 'bottle'
    
    y = get_mvtec_anomaly_classes(dataset_dir+subject+'/test/')
    print(y)
    
    x = get_mvtec_test_images(dataset_dir+subject+'/test/')
    print(x)
    y_hat1 = get_mvtec_gt_filename_counterpart(x[0], dataset_dir+subject+'/ground_truth/')
    y_hat2 = get_mvtec_gt_filename_counterpart(x[-1], dataset_dir+subject+'/ground_truth/')
    
    print(y_hat1)
    print(y_hat2)


def tqdm_test():
    
    for i1 in tqdm(range(5),leave=False):
        print('tante cose belle')
        imsize=(256,256)
        batch_size = 64
        
        print('image size:', imsize)
        print('batch size:', batch_size)
    
        for i2 in tqdm(range(300), leave=False):    
            time.sleep(0.01)
        
        os.system('clear')
     
      
def test_GDE_image_level():

    datamodule = MVTecDatamodule(
      'dataset/bottle/',
      'bottle',
      (256,256),
      64,
      0
    )
    datamodule.setup()
    
    sslm = SSLM.load_from_checkpoint('outputs/computations/bottle/generative_dataset/3-way/best_model.ckpt')
    sslm.eval()
    sslm.to('cuda')
    
    train_embed = []
    for x, _ in datamodule.train_dataloader():
      y_hat, embeddings = sslm(x.to('cuda'))
      embeddings = embeddings.to('cpu')
      train_embed.append(embeddings)
    train_embed = torch.cat(train_embed).to('cpu').detach()

    
    print(train_embed.shape)
    
    test_labels = []
    test_embeds = []
    with torch.no_grad():
        for x, label in datamodule.test_dataloader():
            y_hat, embeddings = sslm(x.to('cuda'))

            # save 
            test_embeds.append(embeddings.to('cpu').detach())
            test_labels.append(label.to('cpu').detach())
    test_labels = torch.cat(test_labels)
    test_embeds = torch.cat(test_embeds)
    
    print(test_embeds.shape)
    print(test_labels.shape)
    test_embeds = torch.nn.functional.normalize(test_embeds, p=2, dim=1)
    train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)
    
    gde = GDE()
    gde.fit(train_embed)
    scores = gde.predict(test_embeds)
    
    print(scores.shape)
    print(scores)

    
    
    int_labels = []
    for x in test_labels:
      if torch.sum(x) == 0:
        int_labels.append(0)
      else:
        int_labels.append(1)
    print(int_labels)
    test_labels = torch.tensor(int_labels)
    
    plot_roc(test_labels, scores, 'bottle')

def test_1d_to_2d():
    x = torch.randn((1,128))

    print(x.shape)
    print(x)
    x1 = np.resize(np.array(x), (int(np.sqrt(x.shape[1])), int(np.sqrt(x.shape[1]))))
    print(x1.shape)
    print(x1)

test_1d_to_2d()
def test_cat():
    x = []
    x.append(torch.randn((1,3,256,256)))
    x.append(torch.randn((1,3,256,256)))
    x.append(torch.randn((1,3,256,256)))
    
    y = torch.cat(x)
    print(y.shape)
    
    
def test_cat2():
    x = []
    x.append(torch.tensor(0))
    x.append(torch.tensor(1))
    x.append(torch.tensor(2))
    
    y = torch.tensor(np.array(x))
    print(y)
    
    a = torch.tensor(3)
    y = y.tolist()
    y.append(a)
    y = torch.tensor(np.array(y))
    print(y)
    
    
def test_cat3():
    y_hat = torch.randn(3,1,128)
    print(y_hat.shape)
    
    y_hat2 = torch.randn(6,1,128)
    
    y = torch.cat([y_hat, y_hat2])
    print(y.shape)


def standardize_and_clip(tensor, min_value=0.0, max_value=1.0):
    """Standardizes and clips input tensor.
    Standardize the input tensor (mean = 0.0, std = 1.0), ensures std is 0.1
    and clips it to values between min/max (default: 0.0/1.0).
    Args:
        tensor (torch.Tensor):
        min_value (float, optional, default=0.0)
        max_value (float, optional, default=1.0)
    Shape:
        Input: :math:`(C, H, W)`
        Output: Same as the input
    Return:
        torch.Tensor (torch.float32): Normalised tensor with values between
            [min_value, max_value]
    """

    tensor = tensor.detach().cpu()

    mean = tensor.mean()
    std = tensor.std()
    if std == 0:
        std += 1e-7

    standardized = tensor.sub(mean).div(std).mul(0.1)
    clipped = standardized.add(0.5).clamp(min_value, max_value)
    return clipped

def test_cam():
    results_dir = 'outputs/computations/'
    defect_type = 'bent'
    subject='grid'
    imsize=(256,256)
    img = Image.open('dataset/'+subject+'/test/'+defect_type+'/000.png').resize(imsize).convert('RGB')
    img = transforms.ToTensor()(img)
    x_query = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
    sslm = md.SSLM.load_from_checkpoint(
        results_dir+subject+'/'+CONST.DEFAULT_CHECKPOINT_MODEL_NAME())
    sslm.eval()
    sslm.set_for_localization(True)
    clone = md.SSLM.load_from_checkpoint(results_dir+subject+'/'+CONST.DEFAULT_CHECKPOINT_MODEL_NAME())
    clone.eval()
    resnet_model_dict = dict(type='resnet18', arch=sslm.model.feature_extractor, layer_name='layer4',input_size=imsize)
    resnet_scorecam = ScoreCAM(resnet_model_dict)
    
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
    #scorecam_map = resnet_scorecam(x_query[None, :].to('cuda'))
    
    
    img = torch.permute(img, (2,1,0))
    #scorecam_map = scorecam_map[0]
    #heatmap = scorecam_map[0].to('cpu')
    heatmap = cv2.resize(heatmap.numpy(), (256,256))
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + 1e-8
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")

    img = cv2.resize(img.numpy(), (256,256))
    numer = img - np.min(img)
    denom = (img.max() - img.min()) + 1e-8
    img = numer / denom
    img = (img * 255).astype("uint8")
    colormap=cv2.COLORMAP_JET
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    output = cv2.addWeighted(heatmap, 0.5, img, 1 - 0.5, 0)
    
    plt.title(y_hat)
    plt.axis('off')
    plt.imshow(output)
    plt.savefig('temp/cam_test.png', bbox_inches='tight')
    
   
def test_cam2():
    results_dir = 'temp/computations/'
    defect_type = 'broken_large'
    subject='bottle'
    imsize=(256,256)
    img = Image.open('dataset/'+subject+'/test/'+defect_type+'/000.png').resize(imsize).convert('RGB')
    img = transforms.ToTensor()(img)
    x_query = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
    sslm = md.SSLM.load_from_checkpoint(
        results_dir+subject+'/'+CONST.DEFAULT_CHECKPOINT_MODEL_NAME())
    sslm.eval()
    sslm.set_for_localization(True)
    clone = md.SSLM.load_from_checkpoint(results_dir+subject+'/'+CONST.DEFAULT_CHECKPOINT_MODEL_NAME())
    clone.eval()
    resnet_model_dict = dict(type='resnet18', arch=sslm.model.feature_extractor, layer_name='layer4',input_size=imsize)
    resnet_scorecam = ScoreCAM(resnet_model_dict)
    
    y_hat, _ = clone(x_query[None, :])
    y_hat = torch.max(y_hat.data, 1)
    y_hat = int(y_hat.indices)
    scorecam_map = resnet_scorecam(x_query[None, :].to('cuda'))
    with torch.no_grad():    
        basic_visualize(
            x_query.cpu(), 
            scorecam_map.type(torch.FloatTensor).cpu(),
            save_path='temp/cam_test.png')

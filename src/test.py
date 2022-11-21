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
    x = torch.randn((1,10))

    print(x.shape)
    print(x)
    x1 = np.resize(np.array(x), (int(np.sqrt(x.shape[1])), int(np.sqrt(x.shape[1]))))
    print(x1.shape)
    print(x1)


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
test_cat3()
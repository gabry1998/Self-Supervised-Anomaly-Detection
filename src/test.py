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
from sklearn import preprocessing

from self_supervised.support.visualization import localize, plot_heatmap, plot_roc


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
    y_hat3 = get_mvtec_gt_filename_counterpart(
        'dataset/bottle/test/good/000.png', 
        dataset_dir+subject+'/ground_truth/')
    print(y_hat1)
    print(y_hat2)
    print(y_hat3)
    
    y_hat4 = get_mvtec_gt_filename_counterpart(
        'dataset/bottle/train/good/000.png', 
        dataset_dir+subject+'/ground_truth/')
    
    print(y_hat4)


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
    
    sslm = SSLM.load_from_checkpoint('outputs/computations/bottle/image_level/best_model.ckpt')
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
    #test_embeds = torch.nn.functional.normalize(test_embeds, p=2, dim=1)
    #train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)
    
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
    
    plot_roc(test_labels, scores)

#test_GDE_image_level()

def test_1d_to_2d():
    x = torch.randn((3249,128))

    print(x.shape)
    print(x)
    x1 = np.resize(np.array(x), (int(np.sqrt(x.shape[0])), int(np.sqrt(x.shape[0]))))
    print(x1.shape)
    print(x1)

#test_1d_to_2d()

def test_gaussian():
    x = [
        [0,0,0,0],
        [0,0,1,0],
        [0,0,1,0],
        [0,0,0,0]
        ]
    x = torch.tensor(x)[None, None, :]
    print(x)
    gs = GaussianSmooth(kernel_size=16, stride=1, device='cpu')
    x1 = gs.upsample(x)
    min_max_scaler = preprocessing.MinMaxScaler()
    x1 = x1.squeeze().squeeze()
    x1 = min_max_scaler.fit_transform(x1)
    
    
    print(x1.shape)
    print(x1)
    plt.imshow(x1)
    plt.savefig('bho.png')


def test_patch_level():
    imsize = (256,256)
    train_img = Image.open('dataset/bottle/test/good/000.png').resize(imsize).convert('RGB')
    test_img = Image.open('dataset/bottle/test/broken_large/001.png').resize(imsize).convert('RGB') 
    
    train_img_tensor = transforms.ToTensor()(train_img)
    train_img_tensor_norm = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(train_img_tensor)
    train_img_tensor_norm = train_img_tensor_norm.unsqueeze(0)
    
    test_img_tensor = transforms.ToTensor()(test_img)
    test_img_tensor_norm = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(test_img_tensor)
    test_img_tensor_norm = test_img_tensor_norm.unsqueeze(0)
    
    sslm = SSLM().load_from_checkpoint('outputs/computations/bottle/patch_level/best_model.ckpt')
    sslm.to('cuda')
    sslm.eval()
    sslm.unfreeze_layers(False)
    
    patches = extract_patches(test_img_tensor_norm, 32, 4)
    
    print('inferencing')
    start = time.time()
    y_hat, embeddings = sslm(patches.to('cuda'))
    y_hat = get_prediction_class(y_hat.to('cpu'))
    print(torch.sum(y_hat)> 0)
    end = time.time() - start
    print('done in', end, 'sec')
    
    print('getting train embeds')
    train_patches = extract_patches(train_img_tensor_norm, 32, 4)
    _, train_embedding = sslm(train_patches.to('cuda'))
    
    gde = GDE()
    gde.fit(train_embedding.to('cpu'))
    print('predicting')
    start = time.time()
    embeddings = embeddings.to('cpu')
    mvtec_test_scores = gde.predict(embeddings)
    end = time.time() - start
    print('done in', end, 'sec')
    
    dim = int(np.sqrt(embeddings.shape[0]))
    out = torch.reshape(mvtec_test_scores, (dim, dim))
    saliency_map_min, saliency_map_max = out.min(), out.max()
    out = (out - saliency_map_min).div(saliency_map_max - saliency_map_min).data
    out[out < 0.35] = 0
    gs = GaussianSmooth(device='cpu')
    out = gs.upsample(out[None, None, :])
    saliency_map_min, saliency_map_max = out.min(), out.max()
    out = (out - saliency_map_min).div(saliency_map_max - saliency_map_min).data

    heatmap = localize(test_img_tensor[None, :], out)
    print(heatmap.min(), heatmap.max())
    image = imagetensor2array(test_img_tensor)
    print(image.min(), image.max())
    heatmap = np.uint8(255 * heatmap)
    image = np.uint8(255 * image)
    plot_heatmap(image, heatmap)
test_patch_level()


def test_reshape():
    gt = get_mvtec_gt_filename_counterpart(
        'dataset/bottle/test/good/000.png', 
        'dataset/bottle/ground_truth')
    gt = ground_truth(gt)

    gt = transforms.ToTensor()(gt).unsqueeze(0)
    print(gt.shape)
    gt_patches = extract_mask_patches(gt, 32, 4)
    print(gt_patches.shape)
    gt_labels = gt2label(gt_patches)
    print(len(gt_labels))
    x = torch.randn((4,3,3))
    print(x)
    print(x.shape)
    num_batch, num_patches, embedding_size = x.shape
    y = torch.reshape(x, (num_patches, num_batch*embedding_size))
    
    x1 = x.flatten()
    x2 = x1.split(embedding_size)
    start = time.time()
    for h in range(num_patches):
        t = torch.tensor([])
        for i in range(len(x2)):
            if (i+h)%num_patches == 0:
                t = torch.cat([t, x2[i]])
        y[h] = t
    end = time.time() - start
    print('done in', end, 'sec')
    print('')
    print(y)
    print(y.shape)
#test_reshape()





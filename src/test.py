from self_supervised.datasets import GenerativeDatamodule
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



def test1():
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


def test2():
    dataset_dir = '/home/ubuntu/TesiAnomalyDetection/dataset/'
    subject = 'toothbrush'
    
    y = get_mvtec_anomaly_classes(dataset_dir+subject+'/test/')
    print(y)
    
    x = get_mvtec_test_images(dataset_dir+subject+'/test/')
    print(x)
    y_hat1 = get_mvtec_gt_filename_counterpart(x[0], dataset_dir+subject+'/ground_truth/')
    y_hat2 = get_mvtec_gt_filename_counterpart(x[-1], dataset_dir+subject+'/ground_truth/')
    
    print(y_hat1)
    print(y_hat2)
 
    
def test3():
    print(CPP.summary)


def test4():
    dataset_dir = '/home/ubuntu/TesiAnomalyDetection/dataset/'
    subject = 'bottle'
    x, y = generate_dataset(
        dataset_dir+subject+'/train/good/',
        classification_task='3-way',
        duplication=True
    )
    x, y = list2np(x, y)
    x, y = np2tensor(x, y)
    print(x.shape, y.shape)
    
    print(x[0])


def test5():
    x = np.array([[1,2],[3,45]])
    print(len(x))


def test6():
    
    for i1 in tqdm(range(5),leave=False):
        print('tante cose belle')
        imsize=(256,256)
        batch_size = 64
        train_val_split = 0.2
        seed = 0
        lr = 0.001
        epochs = 30
        
        print('image size:', imsize)
        print('batch size:', batch_size)
        print('split rate:', train_val_split)
        print('seed:', seed)
        print('optimizer:', 'SGD')
        print('learning rate:', lr)
        print('epochs:', epochs)
    
        for i2 in tqdm(range(300), leave=False):    
            time.sleep(0.01)
        
        os.system('clear')


def generate_rotations(image:Image):
  r90 = image.rotate(90)
  r180 = image.rotate(180)
  r270 = image.rotate(270)
  return image, r90, r180, r270


def generate_patch(image:Image, 
                    area_ratio=(0.02, 0.15), 
                    aspect_ratio=((0.3, 1),(1, 3.3))):
  #print('generate_patch', area_ratio)
  img_area = image.size[0] * image.size[1]
  patch_area = random.uniform(area_ratio[0], area_ratio[1]) * img_area
  patch_aspect = random.choice([random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])])
  patch_w  = int(np.sqrt(patch_area*patch_aspect))
  patch_h = int(np.sqrt(patch_area/patch_aspect))
  org_w, org_h = image.size

  patch_left, patch_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
  patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h
  paste_left, paste_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
  
  return image.crop((patch_left, patch_top, patch_right, patch_bottom)), (paste_left, paste_top)


def paste_patch(image, patch, coords, mask = None):
  aug_image = image.copy()
  aug_image.paste(patch, (coords[0], coords[1]), mask=mask)
  return aug_image


def apply_patch_augmentations(patch:Image, 
                              augmentations:transforms.ColorJitter=None):
  patch = patch.filter(ImageFilter.GaussianBlur(random.randint(0, 2)))
  return augmentations(patch)


def random_color():
  return random.randint(10,240)


def generate_scar(imsize:tuple, 
                  w_range=(2,16), 
                  h_range=(10,25)):
  img_w, img_h = imsize

  #dimensioni sezione
  scar_w = random.randint(w_range[0], w_range[1])
  scar_h = random.randint(h_range[0], h_range[1])

  r = random_color()
  g = random_color()
  b = random_color()

  color = (r,g,b)

  scar = Image.new('RGBA', (scar_w, scar_h), color=color)
  angle = random.randint(-45, 45)
  scar = scar.rotate(angle, expand=True)

  #posizione casuale della sezione
  left, top = random.randint(0, img_w - scar_w), random.randint(0, img_h - scar_h)
  return scar, (left, top)


def get_impaths(main_path):
  return sorted([f for f in glob.glob(main_path+'*.png', recursive = True)])


def load_imgs(main_path, imsize):
  filenames = get_impaths(main_path)
  images = []
  for impath in filenames:
    x = Image.open(impath)
    x = x.resize(imsize)
    images.append(x)
  return images


def extract_patch_embeddings(self, image):
    patches = self.extract_image_patches(image)
    patch_embeddings =[]
    with torch.no_grad():
      for patch in patches:
          logits, patch_embed = self.anomaly.cutpaste_model(patch.to(self.device))
          patch_embeddings.append(patch_embed.to('cpu'))
          del logits, patch

    patch_dim = math.sqrt(len(patches)*self.batch_size)
    patch_matrix = torch.cat(patch_embeddings).reshape(int(patch_dim), int(patch_dim), -1)
    return patch_matrix
      
      
def test7():
    subject:str = 'bottle'
    dataset_type_gen:str = 'generative_dataset'
    classification_task:str = '3-way'
    results_dir = 'outputs/computations/'+subject+'/'+dataset_type_gen+'/'+classification_task+'/'
    model_dir = results_dir+'/best_model.ckpt'
    dataset_dir = 'dataset/'+subject
    sslm = SSLM('3-way')
    sslm = SSLM.load_from_checkpoint(model_dir, model=sslm.model)

    random.seed(0)
    np.random.seed(0)
    print('generating dataset')
    start = time.time()
    datamodule = GenerativeDatamodule(
                dataset_dir,
                classification_task='3-way',
                min_dataset_length=500,
                duplication=True
    )


    datamodule.setup('test')
    end = time.time() - start
    print('generated in '+str(end)+ 'sec')

    x,y = next(iter(datamodule.test_dataloader())) 
    
    patches = extract_patches(x[0], 32, 4)
    
    y_hat, embeddings = sslm(x)

    y_hat = torch.max(y_hat.data, 1)
    y_hat = y_hat.indices
    
    embeddings = embeddings.detach()
    
    print(y_hat.shape)
    print(embeddings.shape)


test7()
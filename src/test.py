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
from sklearn.metrics import roc_curve, auc
import cv2



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

    datamodule = MVTecDatamodule(
      'dataset/bottle/',
      'bottle',
      (256,256),
      64,
      0
    )
    datamodule.setup()
    
    sslm = SSLM.load_from_checkpoint('outputs/computations/bottle/generative_dataset/3-way/best_model.ckpt')
    sslm.to('cuda')
    #x,y = next(iter(datamodule.test_dataloader()))
    
    train_embed = []
    for x, _ in datamodule.train_dataloader():
      y_hat, embeddings = sslm(x.to('cuda'))
      embeddings = embeddings.to('cpu')
      train_embed.append(embeddings)
    train_embed = torch.cat(train_embed).to('cpu').detach()

    
    print(train_embed.shape)
    #y_hat = torch.max(y_hat.data, 1)
    #y_hat = y_hat.indices
    
    
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


def extract_image_patches(image):
    unfold = torch.nn.Unfold((32, 32), stride=4)
    image_patches = unfold(image).squeeze(0).reshape(-1, 3, 32, 32)
    batched_patches = torch.split(image_patches, 128)
    return batched_patches
    #return image_patches

def extract_patch_embeddings(image, sslm):
    patches = extract_image_patches(image)
    patch_embeddings =[]
    with torch.no_grad():
        for patch in patches:
            logits, patch_embed = sslm(patch.to('cuda'))
            patch_embeddings.append(patch_embed.to('cpu'))
            del logits, patch

    patch_dim = math.sqrt(len(patches)*128)
    patch_matrix = torch.cat(patch_embeddings).reshape(int(patch_dim), int(patch_dim), -1)
    return patch_matrix


def test8():
    print('uploading test image')
    imsize = (256,256)
    defect_type = 'good'
    input_image = Image.open('dataset/bottle/test/'+defect_type+'/000.png').resize(imsize).convert('RGB')
    #gt = Image.open('dataset/bottle/ground_truth/good/000_mask.png').resize(imsize)
    sslm = SSLM.load_from_checkpoint(
        'outputs/computations/bottle/generative_dataset/3-way/best_model.ckpt')
    #sslm.to('cuda')
    sslm.eval()
    sslm.model.set_for_localization(True)
    x = transforms.ToTensor()(input_image)
    x = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(x)
    
    #y = transforms.ToTensor()(gt)
    
    x = x[None, :]
    _, l1, l2, l3, l4 = sslm.model.compute_features(x)
    preds, embeds = sslm(x)
    
    pred = torch.argmax(preds).item()
    if pred == 2:
      pred = 1
    #print(pred)
    preds[:, pred].backward()
    g = sslm.model.gradients
    #print(g.shape)
    #print(l4.shape)
    for i in range(512):
        l4[0, i, :, :] *= g[0][i]
    
    l4 = l4.detach()
    
    heatmap = torch.mean(l4, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    
    
    heatmap = heatmap / torch.max(heatmap)
    heatmap = heatmap.numpy()
    
    if pred == 0:
      figtitle = 'good'
    else:  
      figtitle = 'defect'
    
    #print(heatmap.shape)
    plt.title(defect_type+' ('+figtitle+')')
    plt.imshow(heatmap)
    if pred == 1 and defect_type == 'good':
      plt.savefig('gradcam/false_'+defect_type+'_feature.png')
    else:
      plt.savefig('gradcam/'+defect_type+'_feature.png')

    #gs = GaussianSmooth()
    #heatmap = gs.upsample(np.array(torch.tensor(heatmap)[None, :]))
    #print(heatmap.shape)
    #heatmap = heatmap[0]
    heatmap = cv2.resize(heatmap, (input_image.size[1], input_image.size[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + input_image
    superimposed_img = np.uint8(255 * superimposed_img / np.max(superimposed_img))
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    plt.title(defect_type+' ('+figtitle+')')
    plt.imshow(superimposed_img)
    if pred == 1 and defect_type == 'good':
      plt.savefig('gradcam/false_'+defect_type+'_gradcam.png')
    else:
      plt.savefig('gradcam/'+defect_type+'_gradcam.png')


def test9():
    x = torch.randn((3,2,2))
    x = x[None, :]
    print(x.shape)
    gs = GaussianSmooth()
    
    c1 = x[:, 0, :, :]
    c2 = x[:, 1, :, :]
    c3 = x[:, 2, :, :]
    y1 = torch.tensor(gs.upsample(np.array(c1)))
    y2 = torch.tensor(gs.upsample(np.array(c2)))
    y3 = torch.tensor(gs.upsample(np.array(c3)))
    
    out = torch.stack((y1,y2,y3)).permute(1,0,2,3)
    print(out.shape)

test8()
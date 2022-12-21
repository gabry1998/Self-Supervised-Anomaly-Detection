from self_supervised.support.dataset_generator import *
from self_supervised.support.functional import *
from self_supervised.datasets import GenerativeDatamodule
from self_supervised.support.cutpaste_parameters import CPP
from skimage import feature
from skimage.morphology import square, label
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from torchvision import transforms
import collections

def do_patch(img, segmentation=None):
    start = time.time()
    coords = get_random_coordinate(segmentation)
    rotations = [90,180,270]
    temp = img.rotate(random.choice(rotations))
    patch = generate_patch(
        temp, 
        area_ratio=CPP.cutpaste_augmentations['patch']['area_ratio'],
        aspect_ratio=CPP.cutpaste_augmentations['patch']['aspect_ratio'],
        augs=CPP.jitter_transforms)
    patch = patch.filter(ImageFilter.SHARPEN)
    coords, center = check_valid_coordinates_by_container(
        img.size, 
        patch.size, 
        current_coords=coords,
        container_scaling_factor=1.75)
    mask = rect2poly(patch)
    out = paste_patch(img, patch, coords, mask, center)
    end = time.time() - start
    print('patch created in', end, 'sec')
    return out


def do_scar(img, segmentation=None):
    start = time.time()
    #segmentation = obj_mask(img)
    coords = get_random_coordinate(segmentation)
    scar= generate_scar(
            img,
            colorized=True,
            with_padding=True,
            augs=CPP.jitter_transforms
        )
    angle = random.randint(-45,45)
    scar = scar.rotate(angle, expand=True)
    coords, center = check_valid_coordinates_by_container(
        img.size, 
        scar.size, 
        current_coords=coords,
        container_scaling_factor=2.5)
    out = paste_patch(img, scar, coords, scar, center)
    end = time.time() - start
    print('scar created in', end, 'sec')
    return out


def save_fig(my_array, saving_path=None, name='plot.png'):
    if saving_path and not os.path.exists(saving_path):
        os.makedirs(saving_path)
    img = my_array[0]
    shape = img.shape[0]
    hseparator = Image.new(mode='RGB', size=(6,shape), color=(255,255,255))
    my_array = np.hstack([np.hstack(
      [np.array(my_array[i]), np.array(hseparator)]
      ) if i < len(my_array)-1 else np.array(my_array[i]) for i in range(len(my_array))])
    plt.figure(figsize=(30,30))
    plt.imshow(my_array)
    plt.axis('off')
    if saving_path:
        plt.savefig(saving_path+name, bbox_inches='tight')
    else:
        plt.savefig(name, bbox_inches='tight')
    plt.close()
    

def plot_together(good, def1, def2=None, masks=None, saving_path=None, name='plot.png'):
    if saving_path and not os.path.exists(saving_path):
        os.makedirs(saving_path)
    img = good[0]
    shape = img.shape[0]
    hseparator = Image.new(mode='RGB', size=(6,shape), color=(255,255,255))
    
    good_images = np.hstack([np.hstack(
      [np.array(good[i]), np.array(hseparator)]
      ) if i < len(good)-1 else np.array(good[i]) for i in range(len(good))])   
    
    def1 = np.hstack([np.hstack(
      [np.array(def1[i]), np.array(hseparator)]
      ) if i < len(def1)-1 else np.array(def1[i]) for i in range(len(def1))])
    

    def2 = np.hstack([np.hstack(
    [np.array(def2[i]), np.array(hseparator)]
    ) if i < len(def2)-1 else np.array(def2[i]) for i in range(len(def2))])
    
    masks = np.hstack([np.hstack(
      [np.array(masks[i]), np.array(hseparator)]
      ) if i < len(masks)-1 else np.array(masks[i]) for i in range(len(masks))])
    
    vseparator = np.array(Image.new(
      mode='RGB', 
      size=(good_images.shape[1], 6), 
      color=(255,255,255)))
    tot = np.vstack([
        np.vstack([good_images, vseparator]), 
        np.vstack([def1, vseparator]), 
        np.vstack([def2, vseparator]),
        masks])
        
    plt.figure(figsize=(30,30))
    plt.imshow(tot)
    plt.axis('off')
    if saving_path:
        plt.savefig(saving_path+name, bbox_inches='tight')
    else:
        plt.savefig(name, bbox_inches='tight')
    plt.close()


def test_augmentations():
    imsize=(256,256)
    subjects = get_all_subject_experiments('dataset/')
    patch_localization = False
    
    for sub in subjects:
        images = get_image_filenames('dataset/'+sub+'/train/good/')
        goods = []
        masks = []
        patches = []
        scars = []
        for i in range(6):
            img = Image.open(images[i]).resize(imsize).convert('RGB')
            mask = obj_mask(img)
            if patch_localization:
                cropper = Container(img.size, 1.25)
                x = img.crop((cropper.left, cropper.top, cropper.right, cropper.bottom))
                mask = mask.crop((cropper.left, cropper.top, cropper.right, cropper.bottom))
                
                left = random.randint(0,x.size[0]-64)
                top = random.randint(0,x.size[1]-64)
                x = x.crop((left,top, left+64, top+64))
                mask = mask.crop((left,top, left+64, top+64))
                #x = transforms.RandomCrop((64,64))(x)
            else:
                x = img.copy()
                    
            patch = do_patch(img, mask)
            scar = do_scar(img, mask)
            goods.append(np.array(img))
            patches.append(np.array(patch))
            scars.append(np.array(scar))
            masks.append(np.array(mask))
        
        if patch_localization:
            plot_together(goods, patches, scars, masks, 'outputs/dataset_analysis/'+sub+'/', sub+'_artificial_crop.png')
            #save_fig(patches, 'outputs/dataset_analysis/'+sub+'/', sub+'_patch_crop.png')
            #save_fig(scars, 'outputs/dataset_analysis/'+sub+'/', sub+'_scar_crop.png')
            #save_fig(masks, 'outputs/dataset_analysis/'+sub+'/', sub+'_mask_crop.png')
        else:
            plot_together(goods, patches, scars, masks, 'outputs/dataset_analysis/'+sub+'/', sub+'_artificial.png')
            #save_fig(patches, 'outputs/dataset_analysis/'+sub+'/', sub+'_patch.png')
            #save_fig(scars, 'outputs/dataset_analysis/'+sub+'/', sub+'_scar.png')
            #save_fig(masks, 'outputs/dataset_analysis/'+sub+'/', sub+'_mask.png')
        os.system('clear')
     

def check_all_subject():
    subjects = get_all_subject_experiments('dataset/')
    imsize=(256,256)
    patch_localization = False
    
    goods = []
    masks = []
    patches = []
    scars = []
    for subject in subjects:
        img = Image.open('dataset/'+subject+'/train/good/005.png').resize(imsize).convert('RGB')
        mask = obj_mask(img)
        if patch_localization:
            cropper = Container(img.size, 1.25)
            x = img.crop((cropper.left, cropper.top, cropper.right, cropper.bottom))
            mask = mask.crop((cropper.left, cropper.top, cropper.right, cropper.bottom))
            
            left = random.randint(0,x.size[0]-64)
            top = random.randint(0,x.size[1]-64)
            x = x.crop((left,top, left+64, top+64))
            mask = mask.crop((left,top, left+64, top+64))
            #x = transforms.RandomCrop((64,64))(x)
            
        else:
            x = img.copy()
        patch = do_patch(x, mask)
        scar = do_scar(x, mask)
        goods.append(np.array(x))
        patches.append(np.array(patch))
        scars.append(np.array(scar))
        masks.append(np.array(mask))
    
    if patch_localization:
        plot_together(goods, patches, scars, masks, 'outputs/dataset_analysis/', 'artificial_overall_crop.png')
    else:
        plot_together(goods, patches, scars, masks, 'outputs/dataset_analysis/', 'artificial_overall.png')

    os.system('clear')


def check_distribution():
    artificial = GenerativeDatamodule(
        'dataset/bottle/',
        (256,256),
        256,
        duplication=True,
        polygoned=True,
        colorized_scar=True
    )
    artificial.setup('test')
    labels = artificial.test_dataset.labels
    c = collections.Counter(labels)
    instances = [c[x] for x in sorted(c.keys())]
    keys = sorted(c.keys())
    font_title = {
        'weight': 'bold',
        'size': 22
    }
    font = {
        'size': 22
    }
    plt.figure(figsize=(20,20))
    plt.bar(keys, instances)
    
    plt.title('Dataset distribution', fontdict=font_title)
    plt.ylabel('Frequency', fontdict=font)
    plt.xticks([0,1,2], ['good', 'polygon patch', 'colored scar'], font=font)
    plt.savefig('distribution.png', bbox_inches='tight')
    
test_augmentations()
check_all_subject()
#check_distribution()



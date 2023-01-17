from tqdm import tqdm
from self_supervised.support.dataset_generator import *
from self_supervised.support.functional import *
from self_supervised.datasets import GenerativeDatamodule
from self_supervised.support.cutpaste_parameters import CPP
from torchvision import transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import collections
import cv2 


def save_fig2(my_array, saving_path=None, name='plot.png'):
    if saving_path and not os.path.exists(saving_path):
        os.makedirs(saving_path)
    
    if len(my_array) > 1:
        fig, axs = plt.subplots(1,len(my_array), figsize=(30,30))
        for i in range(len(my_array)):
            axs[i].imshow(my_array[i])
            axs[i].axis('off')
    else:
        plt.imshow(my_array[0])
        plt.axis('off')
        
    if saving_path:
        plt.savefig(saving_path+name, bbox_inches='tight')
    else:
        plt.savefig(name, bbox_inches='tight')
    plt.close()


def do_patch(img, original=None, segmentation=None, subject=None, patch_loc=False):
    segmentation = np.array(segmentation.convert('1'))
    coordinates = np.flip(np.column_stack(np.where(segmentation == 1)), axis=1)
    coords = get_random_coordinate(coordinates)
    aspect_ratio = CPP.cutpaste_augmentations['patch']['aspect_ratio']
    classes = get_all_subject_experiments('dataset/')
    idx = np.where(classes == subject)
    random_subject = random.choice(np.delete(classes, idx))
    cutting = Image.open('dataset/'+random_subject+'/train/good/000.png').resize(img.size).convert('RGB')
    if subject in np.array(['carpet','grid','leather','tile','wood']):
        patch = generate_patch(
            cutting,
            area_ratio=[0.02, 0.05],
            aspect_ratio=aspect_ratio,
            #augs=CPP.jitter_transforms,
            colorized=False)
    else:
        patch = generate_patch(
            original,
            area_ratio=[0.02, 0.09],
            aspect_ratio=aspect_ratio,
            #augs=CPP.jitter_transforms,
            colorized=False)
    coords, _ = check_valid_coordinates_by_container(
        img.size, 
        patch.size, 
        current_coords=coords,
        container_scaling_factor=2
    )
    mask = None
    right = 1
    left = 1
    top = 1
    bottom = 1
    new_width = patch.size[0] + right + left
    new_height = patch.size[1] + top + bottom
    new_img = Image.new(img.mode, (new_width, new_height), color='black')
    mask = rect2poly(patch, regular=False, sides=8)
    new_img = paste_patch(new_img, patch, (1,1), mask)
    new_mask = paste_patch(new_img, mask, (1,1), None)
    out = paste_patch(img, patch, coords, mask)
    return patch, new_img, new_mask, out, cutting



def do_scar(img, original=None, segmentation=None, subject=None, patch_loc=False):
    classes = np.array(get_all_subject_experiments('dataset/'))
    idx = np.where(classes == subject)
    
    if subject in np.array(['carpet','grid','leather','tile','wood']):
        random_subject = random.choice(np.delete(classes, idx))
        cutting = Image.open('dataset/'+random_subject+'/train/good/000.png').resize(img.size).convert('RGB')
        scar= generate_scar(
            cutting,
            colorized=False,
            w_range=[2,16],
            h_range=[10,25],
            #augs=CPP.jitter_transforms,
            color_type='average' # random, average, sample
        )
    else:
        scar= generate_scar(
            original,
            colorized=False,
            w_range=[2,16],
            h_range=[10,25],
            #augs=CPP.jitter_transforms,
            color_type='average' # random, average, sample
        )
    angle = random.randint(-45,45)
    scar = scar.rotate(angle, expand=True)
    scar = paste_patch(scar, scar, (0,0), scar)
    return scar


def check():
    imsize=(128,128)
    subjects = get_all_subject_experiments('dataset/')
    for i in tqdm(range(len(subjects))):
        sub = subjects[i]
        patches = []
        polygons = []
        masked_polys = []
        lines = []
        og = []
        images = get_image_filenames('dataset/'+sub+'/train/good/')
        for i in range(6):
            img = Image.open(images[i]).resize(imsize).convert('RGB')
            segm = obj_mask(img)
            x = img.copy()
            #x = CPP.jitter_transforms(x)
            og_patch, patch, mask, out, other_im = do_patch(x,img, segm, sub, False)
            scar = do_scar(x,img, None, sub, False)
            og.append(x)
            patches.append(og_patch)
            polygons.append(np.array(patch))
            masked_polys.append(np.array(mask))
            lines.append(np.array(scar))
        save_fig2([other_im], saving_path='brutta_copia/transformations/'+sub+'/', name=sub+'_other_image.png')
        save_fig2([x, og_patch, mask, patch, out], saving_path='brutta_copia/transformations/'+sub+'/', name=sub+'_transform_chain.png')
        save_fig2(patches, saving_path='brutta_copia/transformations/'+sub+'/', name=sub+'_patch.png')
        save_fig2(og, saving_path='brutta_copia/transformations/'+sub+'/', name=sub+'_original.png')
        save_fig2(polygons, saving_path='brutta_copia/transformations/'+sub+'/', name=sub+'_polygon.png')
        save_fig2(masked_polys, saving_path='brutta_copia/transformations/'+sub+'/', name=sub+'_mask.png')
        save_fig2(lines, saving_path='brutta_copia/transformations/'+sub+'/', name=sub+'_scar.png')
        
check()
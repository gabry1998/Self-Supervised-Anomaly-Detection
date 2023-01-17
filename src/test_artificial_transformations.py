from tqdm import tqdm
from skimage.segmentation import slic
from skimage import color
from self_supervised.support.dataset_generator import *
from self_supervised.support.functional import *
from self_supervised.datasets import GenerativeDatamodule
from self_supervised.support.cutpaste_parameters import CPP
from skimage.measure import regionprops
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import collections
import cv2 



def do_patch(img, image_for_cutting=None, segmentation=None, patch_loc=False):
    factor = 1 if patch_loc else 2
    segmentation = np.array(segmentation.convert('1'))
    coordinates = np.flip(np.column_stack(np.where(segmentation == 1)), axis=1)
    coords = get_random_coordinate(coordinates)
    image_for_cutting = image_for_cutting.rotate(random.choice([90,180,270]))
    patch = generate_patch(
        image_for_cutting,
        area_ratio=CPP.rectangle_area_ratio,
        aspect_ratio=CPP.rectangle_aspect_ratio,
        augs=CPP.jitter_transforms
    )
    
    if check_patch_and_defect_similarity(img, patch) > 0.999:
        patch = ImageOps.invert(patch)
    coords, _ = check_valid_coordinates_by_container(
        img.size, 
        patch.size, 
        current_coords=coords,
        container_scaling_factor=factor
    )
    mask = None
    mask = rect2poly(patch, regular=False, sides=8)
    
    x = paste_patch(img, patch, coords, mask)
    return x


def do_scar(img, image_for_cutting=None, segmentation=None, patch_loc=False):
    factor = 1 if patch_loc else 2.5
    segmentation = np.array(segmentation.convert('1'))
    coordinates = np.flip(np.column_stack(np.where(segmentation == 1)), axis=1)
    coords = get_random_coordinate(coordinates)
    image_for_cutting = image_for_cutting.rotate(random.choice([90,180,270]))
    scar= generate_patch(
        image_for_cutting,
        area_ratio=CPP.scar_area_ratio,
        aspect_ratio=CPP.scar_aspect_ratio,
        #augs=CPP.jitter_transforms
    )
    if check_patch_and_defect_similarity(img, scar) > 0.99:
        scar = ImageOps.invert(scar)
    angle = random.randint(-45,45)
    scar = scar.convert('RGBA')
    scar = scar.rotate(angle, expand=True)
    coords, _ = check_valid_coordinates_by_container(
        img.size, 
        scar.size, 
        current_coords=coords,
        container_scaling_factor=factor
    )
    
    x = paste_patch(img, scar, coords, scar)
    return x


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
    
    if def2 is not None:
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
    if def2 is not None:
        tot = np.vstack([
            np.vstack([good_images, vseparator]), 
            np.vstack([def1, vseparator]), 
            np.vstack([def2, vseparator]),
            masks])
    else:
        tot = np.vstack([
            np.vstack([good_images, vseparator]), 
            np.vstack([def1, vseparator]), 
            masks])
        
    plt.figure(figsize=(30,30))
    plt.imshow(tot)
    plt.axis('off')
    if saving_path:
        plt.savefig(saving_path+name, bbox_inches='tight')
    else:
        plt.savefig(name, bbox_inches='tight')
    plt.close()


def get_superpixels(image, segm):
    image_array = np.array(image)
    segments = slic(image_array, n_segments = segm, sigma = 5, convert2lab=True)
    superpixels = color.label2rgb(segments, image_array, kind='avg')
    return superpixels

def test_augmentations(patch_localization = False):
    imsize=(256,256)
    patchsize = 32
    subjects = get_all_subject_experiments('dataset/')
    classes = get_all_subject_experiments('dataset/')
    for i in tqdm(range(len(subjects))):
        sub = subjects[i]
        images = get_image_filenames('dataset/'+sub+'/train/good/')
        goods = []
        masks = []
        patches = []
        scars = []
        # fixed mask
        if sub in np.array(['carpet','grid','leather','tile','wood']):
            fixed_segmentation = Image.new(size=imsize, mode='RGB', color='white')
        else:
            fixed_segmentation = obj_mask(Image.open('dataset/'+sub+'/train/good/000.png').resize(imsize).convert('RGB'))
        
        for i in tqdm(range(6)):
            img = Image.open(images[i]).resize(imsize).convert('RGB')
            
            # mask geneneration
            if sub in np.array(['hazelnut', 'screw', 'metal_nut']):
                mask = obj_mask(img)
            else:
                mask = fixed_segmentation
            
            # image for cut
            if sub in np.array(['carpet','grid','leather','tile','wood']):
                idx = np.where(classes == sub)
                random_subject = random.choice(np.delete(classes, idx))
                cutting = Image.open('dataset/'+random_subject+'/train/good/000.png').resize(imsize).convert('RGB')
            else:
                cutting = img.copy()
            
            # crop if patch localization
            if patch_localization:
                left = random.randint(0,img.size[0]-patchsize)
                top = random.randint(0,img.size[1]-patchsize)
                x = img.crop((left,top, left+patchsize, top+patchsize))
                mask = mask.crop((left,top, left+patchsize, top+patchsize))
                cutting = transforms.RandomCrop(patchsize)(cutting)
            else:
                x = img.copy()
            if torch.sum(transforms.ToTensor()(mask)) > int((patchsize*patchsize)/2):
                patch = do_patch(x,image_for_cutting=cutting, segmentation=mask, patch_loc=patch_localization)
                scar = do_scar(x,image_for_cutting=cutting, segmentation=mask, patch_loc=patch_localization)
            else:
                patch = x.copy()
                scar = x.copy()
            goods.append(np.array(x))
            patches.append(np.array(patch))
            scars.append(np.array(scar))
            masks.append(np.array(mask))
        
        if patch_localization:
            plot_together(goods, patches, scars, masks, 'brutta_copia/dataset_analysis/'+sub+'/', sub+'_artificial_crop.png')
        else:
            plot_together(goods, patches, scars, masks, 'brutta_copia/dataset_analysis/'+sub+'/', sub+'_artificial.png')
        os.system('clear')
     

def check_all_subject(patch_localization = False):
    subjects = get_all_subject_experiments('dataset/')
    imsize=(256,256)
    patchsize = 32
    goods = []
    masks = []
    patches = []
    scars = []
    classes = get_all_subject_experiments('dataset/')
    
    
    for i in tqdm(range(len(subjects))):
        subject = subjects[i]
        img = Image.open('dataset/'+subject+'/train/good/005.png').resize(imsize).convert('RGB') 
           
        # mask generation    
        if subject in np.array(['carpet','grid','leather','tile','wood']):
            mask = Image.new(size=imsize, mode='RGB', color='white')
        else:
            mask = obj_mask(img)

        # image for cut
        if subject in np.array(['carpet','grid','leather','tile','wood']):
            idx = np.where(classes == subject)
            random_subject = random.choice(np.delete(classes, idx))
            cutting = Image.open('dataset/'+random_subject+'/train/good/000.png').resize(imsize).convert('RGB')
        else:
            cutting = img.copy()
        
        # patch loc -> crop
        if patch_localization:
            left = random.randint(0,img.size[0]-patchsize)
            top = random.randint(0,img.size[1]-patchsize)
            x = img.crop((left,top, left+patchsize, top+patchsize))
            mask = mask.crop((left,top, left+patchsize, top+patchsize))
            cutting = transforms.RandomCrop(patchsize)(cutting)
        else:
            x = img.copy()
        if torch.sum(transforms.ToTensor()(mask)) > int((patchsize*patchsize)/2):
            patch = do_patch(x,image_for_cutting=cutting, segmentation=mask, patch_loc=patch_localization)
            scar = do_scar(x,image_for_cutting=cutting, segmentation=mask, patch_loc=patch_localization)
        else:
            patch = x.copy()
            scar = x.copy()
        goods.append(np.array(x))
        patches.append(np.array(patch))
        scars.append(np.array(scar))
        masks.append(np.array(mask))
    
    if patch_localization:
        plot_together(goods, patches, scars, masks, 'brutta_copia/dataset_analysis/', 'artificial_overall_crop.png')
    else:
        plot_together(goods, patches, scars, masks, 'brutta_copia/dataset_analysis/', 'artificial_overall.png')

    os.system('clear')


def check_distribution():
    artificial = GenerativeDatamodule(
        'bottle',
        'dataset/bottle/',
        (256,256),
        256,
        duplication=True,
        polygoned=True,
        colorized_scar=True
    )
    artificial.setup('test')
    _, labels, _ = next(iter(artificial.test_dataloader()))
    print(labels)
    c = collections.Counter(np.array(labels))
    print(c)
    instances = [c[x] for x in sorted(c.keys())]
    print(instances)
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
    plt.xticks([0,1,2], ['good', 'polygon patch','scar'], font=font)
    plt.savefig('distribution.png', bbox_inches='tight')
    
test_augmentations(True)
check_all_subject(True)
#check_distribution()




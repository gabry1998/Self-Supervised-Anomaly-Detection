import random
from PIL import Image, ImageFilter
import numpy as np
from .cutpaste_parameters import CPP
from .functional import get_image_filenames, duplicate_filenames

def generate_rotations(image):
    r90 = image.rotate(90)
    r180 = image.rotate(180)
    r270 = image.rotate(270)
    return image, r90, r180, r270


def generate_rotation(image):
    rotation = random.choice([0, 90, 180, 270])
    return image.rotate(rotation)


def generate_patch(
        image, 
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


def paste_patch(image, patch, coords, mask=None):
    aug_image = image.copy()
    aug_image.paste(patch, (coords[0], coords[1]), mask=mask)
    return aug_image


def apply_jittering(img, augmentations):
    return augmentations(img)

# not used
def apply_gaussian_blur(img):
    return img.filter(ImageFilter.BoxBlur(random.randint(0, 3)))


def random_color():
    return random.randint(10,240)


def generate_scar(imsize:tuple, w_range=(2,16), h_range=(10,25)):
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


def generate_dataset(
        dataset_dir:str, 
        imsize=(256,256),
        classification_task:str='binary',
        min_dataset_length:int=2000,
        duplication=False):
    
    raw_images_filenames = get_image_filenames(dataset_dir) # qualcosa come ../dataset/bottle/train/good/
    if duplication:
        raw_images_filenames = duplicate_filenames(
            raw_images_filenames,
            min_dataset_length
        )
    
    length = raw_images_filenames.shape[0]
    if classification_task == '3-way':
        labels = np.array(np.random.uniform(0,3, length), dtype=int)
    if classification_task == 'binary':
        labels = np.array(np.random.uniform(0,2, length), dtype=int)
    else:
        exit(1)
    
    data = []
    for i in range(length):
        x = raw_images_filenames[i]
        y = labels[i]
        if classification_task == '3-way':
            x = generate_cutpaste_3way(x, y)
        if classification_task == 'binary':
            x = generate_cutpaste_binary(x, y)
        data.append(x)
    
    data = np.array(data)
    return data, labels
    
def generate_cutpaste_3way(x, y, imsize=(256,256)):
    x = Image.open(x).resize(imsize).convert('RGB')
    x = generate_rotation(x)
    if y == 0:
        return x
    if y == 1:
        patch, coords = generate_patch(
            x, 
            CPP.cutpaste_augmentations['patch']['area_ratio'], 
            CPP.cutpaste_augmentations['patch']['aspect_ratio'])
        patch = apply_jittering(patch, CPP.jitter_transforms)
        x = paste_patch(x, patch, coords)
        return x
    if y == 2:
        patch, coords = generate_scar(
            x.size, 
            CPP.cutpaste_augmentations['scar']['width'], 
            CPP.cutpaste_augmentations['scar']['thiccness'])
        x = paste_patch(x, patch, coords, patch)
        return x
    

def generate_cutpaste_binary(x, y, imsize=(256,256)):
        x = Image.open(x).resize(imsize).convert('RGB')
        x = generate_rotation(x)
        if y == 0:
            return x
        else:
            if random.randint(0,1) == 1:
                patch, coords = generate_patch(
                    x, 
                    CPP.cutpaste_augmentations['patch']['area_ratio'], 
                    CPP.cutpaste_augmentations['patch']['aspect_ratio'])
                patch = apply_jittering(patch, CPP.jitter_transforms)
                x = paste_patch(x, patch, coords)
                return x
            else:
                patch, coords = generate_scar(
                    x.size, 
                    CPP.cutpaste_augmentations['scar']['width'], 
                    CPP.cutpaste_augmentations['scar']['thiccness'])
                x = paste_patch(x, patch, coords, patch)
                return x

    
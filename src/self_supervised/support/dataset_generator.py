import random
from PIL import Image, ImageFilter
import numpy as np
from .cutpaste_parameters import CPP
from .functional import get_image_filenames, duplicate_filenames
import time

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
        classification_task:str='binary'):
    
    raw_images_filenames = get_image_filenames(dataset_dir) # qualcosa come ../dataset/bottle/train/good/
    data = []
    labels = []
    
    augs = CPP.jitter_transforms
    area_ratio_patch = CPP.cutpaste_augmentations['patch']['area_ratio']
    aspect_ratio_patch = CPP.cutpaste_augmentations['patch']['aspect_ratio']
    scar_width = CPP.cutpaste_augmentations['scar']['width']
    scar_thiccness = CPP.cutpaste_augmentations['scar']['thiccness']
    
    print('generating dataset', '('+str(len(raw_images_filenames))+' filenames)')
    start = time.time()
    for filename in raw_images_filenames:
        image = Image.open(filename).resize(imsize).convert('RGB')
        r0, r90, r180, r270 = generate_rotations(image)
        rotations = [r0, r90, r180, r270]

        for img in rotations:
            data.append(img)
            labels.append(0)
        
            if classification_task=='binary':
                if random.randint(0,1)==1:
                    #cutpaste
                    x, coords = generate_patch(img, area_ratio_patch, aspect_ratio_patch)
                    x = apply_jittering(x, augs)
                    new_img = paste_patch(img, x, coords)
                    data.append(new_img)
                    labels.append(1)
                else:
                    #scar
                    x, coords = generate_scar(img.size, scar_width, scar_thiccness)
                    new_img = paste_patch(img, x, coords, x)
                    data.append(new_img)
                    labels.append(1)
                
            if classification_task=='3-way':
                #cutpaste
                x, coords = generate_patch(img, area_ratio_patch, aspect_ratio_patch)
                x = apply_jittering(x, augs)
                new_img = paste_patch(img, x, coords)
                data.append(new_img)
                labels.append(1)
                #scar
                x, coords = generate_scar(img.size, scar_width, scar_thiccness)
                new_img = paste_patch(img, x, coords, x)
                data.append(new_img)
                labels.append(2)
    end = time.time() - start
    print('done generation in ', str(end), 'sec')
    
    return data, labels
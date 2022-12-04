import random
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import numpy as np
from .cutpaste_parameters import CPP
from .functional import get_image_filenames, duplicate_filenames
import time



class Deformer:
    def __init__(self, img_size, area_ratio=(0.02, 0.15), aspect_ratio=((0.3, 1),(1, 3.3))) -> None:
        self.img_size = img_size

        img_area = self.img_size[0] * self.img_size[1]
        patch_area = random.uniform(area_ratio[0], area_ratio[1]) * img_area
        patch_aspect = random.choice([random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])])
        patch_w  = int(np.sqrt(patch_area*patch_aspect))
        patch_h = int(np.sqrt(patch_area/patch_aspect))
        org_w, org_h = self.img_size
        self.crop_left, self.crop_top = random.randint(0, (org_w - patch_w)), random.randint(0, (org_h - patch_h))
        self.crop_right, self.crop_bottom = self.crop_left + patch_w, self.crop_top + patch_h
        self.paste_left, self.paste_top = random.randint(0, (org_w - patch_w)), random.randint(0, (org_h - patch_h))
    def getmesh(self, img):
        return [(
                # target rectangle
                (self.crop_left, self.crop_top,self.crop_right, self.crop_bottom),
                # corresponding source quadrilateral
                (np.random.randint(0, self.img_size[0]), 
                 np.random.randint(0,self.img_size[0]),
                 np.random.randint(0,self.img_size[0]),
                 np.random.randint(0,self.img_size[0]),
                 np.random.randint(0,self.img_size[0]),
                 np.random.randint(0,self.img_size[0]),
                 np.random.randint(0,self.img_size[0]),
                 np.random.randint(0,self.img_size[0]))
                )]


def generate_rotations(image):
    r90 = image.rotate(90)
    r180 = image.rotate(180)
    r270 = image.rotate(270)
    return image, r90, r180, r270


def generate_rotation(image):
    rotation = random.choice([0, 90, 180, 270])
    return image.rotate(rotation)


def get_random_points(width, height, num_points=3):
    points = []
    for _ in range(num_points):
        points.append((random.randint(0, width), random.randint(0, height)))
    return points


def generate_patch_distorted(
        image, 
        area_ratio=(0.02, 0.15),
        aspect_ratio=((0.3, 1),(1, 3.3))):
    sd = Deformer(
        image.size, 
        area_ratio, 
        aspect_ratio)
    deformed = ImageOps.deform(image, sd)
    return deformed.crop((sd.crop_left, sd.crop_top, sd.crop_right, sd.crop_bottom)), (sd.paste_left, sd.paste_top)

def generate_patch(
        image, 
        area_ratio=(0.02, 0.15), 
        aspect_ratio=((0.3, 1),(1, 3.3)),
        polygoned=True):

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

    mask = None
    if polygoned:
        mask = Image.new('RGBA', (patch_w, patch_h), (0, 0, 0, 0)) 
        draw = ImageDraw.Draw(mask)
            
        points = get_random_points(mask.size[0], mask.size[1], random.randint(3,5))
        
        draw.polygon(points, fill='white')
    
    return image.crop((patch_left, patch_top, patch_right, patch_bottom)), mask, (paste_left, paste_top)


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


def generate_scar_new(image, w_range=(2,16), h_range=(10,25), augs=None):
    img_w, img_h = image.size
    right = 1
    left = 1
    top = 1
    bottom = 1

    scar_w = random.randint(w_range[0], w_range[1])
    scar_h = random.randint(h_range[0], h_range[1])
    new_width = scar_w + right + left
    new_height = scar_h + top + bottom
    patch_left, patch_top = random.randint(0, img_w - scar_w), random.randint(0, img_h - scar_h)
    patch_right, patch_bottom = patch_left + scar_w, patch_top + scar_h
    
    scar = image.crop((patch_left, patch_top, patch_right, patch_bottom))
    scar_with_pad = Image.new(image.mode, (new_width, new_height), (255, 255, 255))
    scar = apply_jittering(scar, augs)
    scar_with_pad.paste(scar, (left, top))
    scar = scar_with_pad.convert('RGBA')
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
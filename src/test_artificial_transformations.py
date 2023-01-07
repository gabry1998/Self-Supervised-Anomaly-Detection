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


def apply_wrinkle(img, wrinkles):
    img = np.array(img).astype("float32") / 255.0
    wrinkles = np.array(wrinkles.convert('L')).astype("float32") / 255.0
    # apply linear transform to stretch wrinkles to make shading darker
    # C = A*x+B
    # x=1 -> 1; x=0.25 -> 0
    # 1 = A + B
    # 0 = 0.25*A + B
    # Solve simultaneous equations to get:
    # A = 1.33
    # B = -0.33
    wrinkles = 1.33 * wrinkles -0.33

    # threshold wrinkles and invert
    thresh = cv2.threshold(wrinkles,0.5,1,cv2.THRESH_BINARY)[1]
    thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR) 
    thresh_inv = 1-thresh

    # shift image brightness so mean is mid gray
    mean = np.mean(wrinkles)
    shift = mean - 0.5
    wrinkles = cv2.subtract(wrinkles, shift)

    # convert wrinkles from grayscale to rgb
    wrinkles = cv2.cvtColor(wrinkles,cv2.COLOR_GRAY2BGR) 

    # do hard light composite and convert to uint8 in range 0 to 255
    # see CSS specs at https://www.w3.org/TR/compositing-1/#blendinghardlight
    low = 2.0 * img * wrinkles
    high = 1 - 2.0 * (1-img) * (1-wrinkles)
    result = ( 255 * (low * thresh_inv + high * thresh) ).clip(0, 255).astype(np.uint8)
    return Image.fromarray(result).convert('RGB')


def generate_wrinkle(patch):
    wrinkle = Image.open('wrinkled.png').convert('RGBA')
    wrinkle_left = random.randint(0, wrinkle.size[0] - patch.width) 
    wrinkle_top = random.randint(0, wrinkle.size[1] - patch.height)
    wrinkle_right, wrinkle_bottom = wrinkle_left + patch.width, wrinkle_top + patch.height
    wrinkle = wrinkle.crop((wrinkle_left, wrinkle_top, wrinkle_right, wrinkle_bottom))
    return wrinkle


def check_valid_coords(center, imsize, patchsize):
    width, height = imsize
    patch_width, patch_height = patchsize
    patch_left = center[0] - int(patch_width/2)
    patch_top = center[1] - int(patch_height/2)
    if patch_left < 0:
        patch_left = int(width/2) - patch_width
    if patch_top < 0:
        patch_top = int(height/2) - patch_height
         
    patch_right = patch_left + patch_width
    patch_bottom = patch_top + patch_height
    if patch_right > width:
        patch_left = int(width/2) - patch_width
    if patch_bottom > height:
        patch_top = int(height/2) - patch_height
    return (patch_left, patch_top, patch_right, patch_bottom)


def do_patch(img, original=None, segmentation=None, subject=None, patch_loc=False):
    factor = 2
    if patch_loc:
        factor = 1
    #segmentation = obj_mask(img)
    coords = get_random_coordinate(segmentation)
    area_ratio = CPP.cutpaste_augmentations['patch']['area_ratio']
    aspect_ratio = CPP.cutpaste_augmentations['patch']['aspect_ratio']
    start = time.time()
    x = img.copy()
    classes = get_all_subject_experiments('dataset/')
    idx = np.where(classes == subject)
    random_subject = random.choice(np.delete(classes, idx))
    cutting = Image.open('dataset/'+random_subject+'/train/good/000.png').resize((256,256)).convert('RGB')
    if subject in np.array(['carpet','grid','leather','tile','wood']):
        patch = generate_patch(
            cutting,
            area_ratio=[0.02, 0.05],
            aspect_ratio=aspect_ratio,
            augs=CPP.jitter_transforms,
            colorized=False)
    else:
        patch = generate_patch(
            original,
            area_ratio=[0.02, 0.09],
            aspect_ratio=aspect_ratio,
            augs=CPP.jitter_transforms,
            colorized=False)
    coords, _ = check_valid_coordinates_by_container(
        img.size, 
        patch.size, 
        current_coords=coords,
        container_scaling_factor=factor
    )
    mask = None
    mask = rect2poly(patch, regular=False, sides=8)
    x = paste_patch(x, patch, coords, mask)
    end = time.time() - start
    print('patch created in', end, 'sec')
    return x


def do_scar(img, original=None, segmentation=None, subject=None, patch_loc=False):
    factor = 2.5
    if patch_loc:
        factor = 1
    start = time.time()
    coords = get_random_coordinate(segmentation)
    classes = np.array(get_all_subject_experiments('dataset/'))
    x = img.copy()
    idx = np.where(classes == subject)
    
    if subject in np.array(['carpet','grid','leather','tile','wood']):
        random_subject = random.choice(np.delete(classes, idx))
        cutting = Image.open('dataset/'+random_subject+'/train/good/000.png').resize((256,256)).convert('RGB')
        scar= generate_scar(
            cutting,
            colorized=False,
            augs=CPP.jitter_transforms,
            color_type='average' # random, average, sample
        )
    else:
        scar= generate_scar(
            original,
            colorized=False,
            augs=CPP.jitter_transforms,
            color_type='average' # random, average, sample
        )
    coords, _ = check_valid_coordinates_by_container(
        img.size, 
        scar.size, 
        current_coords=coords,
        container_scaling_factor=factor
    )
    angle = random.randint(-45,45)
    scar = scar.rotate(angle, expand=True)
    x = paste_patch(x, scar, coords, scar)
    
    end = time.time() - start
    print('scar created in', end, 'sec')
    return x


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


def test_augmentations(patch_localization = False):
    imsize=(256,256)
    subjects = get_all_subject_experiments('dataset/')
    
    
    for sub in subjects:
        images = get_image_filenames('dataset/'+sub+'/train/good/')
        goods = []
        masks = []
        patches = []
        scars = []
        fixed_segmentation = obj_mask(Image.open('dataset/'+sub+'/train/good/000.png').resize(imsize).convert('RGB'))
        for i in range(6):
            img = Image.open(images[i]).resize(imsize).convert('RGB')
            
            if sub in np.array(['hazelnut', 'screw', 'metal_nut']):
                mask = obj_mask(img)
            else:
                mask = fixed_segmentation
            if patch_localization:
                x = img.copy()
                left = random.randint(0,x.size[0]-64)
                top = random.randint(0,x.size[1]-64)
                x = x.crop((left,top, left+64, top+64))
                mask = mask.crop((left,top, left+64, top+64))
            else:
                x = img.copy()
                
            x = CPP.jitter_transforms(x)
            if torch.sum(transforms.ToTensor()(mask)) > 0:
                patch = do_patch(x,img, mask, sub, patch_localization)
                scar = do_scar(x,img, mask, sub, patch_localization)
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
    
    goods = []
    masks = []
    patches = []
    scars = []
    for subject in subjects:
        img = Image.open('dataset/'+subject+'/train/good/005.png').resize(imsize).convert('RGB')
        
        mask = obj_mask(img)
        if patch_localization:
            left = random.randint(0,img.size[0]-64)
            top = random.randint(0,img.size[1]-64)
            x = img.crop((left,top, left+64, top+64))
            mask = mask.crop((left,top, left+64, top+64))
        else:
            x = img.copy()
        x = CPP.jitter_transforms(x)
        if torch.sum(transforms.ToTensor()(mask)) > 0:
            patch = do_patch(x,img, mask, subject, patch_localization)
            scar = do_scar(x,img, mask, subject, patch_localization)
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




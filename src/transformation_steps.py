from self_supervised import dataset_generator as gntr
from self_supervised import functional as f
from self_supervised.datasets import CPP
from torchvision import transforms
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tqdm import tqdm

def show_steps():
    imsize=(256,256)
    patch_size=64
    subject = 'bottle'
    
    images = f.get_filenames('dataset/'+subject+'/train/good/')
    classes = f.get_all_subject_experiments('dataset/')

    for i in tqdm(range(len(classes))):
        sub = classes[i]
        saving_path='brutta_brutta_copia/dataset_analysis/'+sub+'/transformation_steps/'
        
        # load im
        image = Image.open('dataset/'+sub+'/train/good/000.png').resize(imsize).convert('RGB')
        
        # mask gen
        if subject in np.array(['carpet','grid','leather','tile','wood']):
            segmentation = Image.new(size=imsize, mode='RGB', color='white')
        else:
            segmentation = gntr.obj_mask(image)
            
        # img for cut
        if subject in np.array(['carpet','grid','leather','tile','wood']):
            idx = np.where(classes == subject)
            random_subject = random.choice(np.delete(classes, idx))
            image_for_cutting = Image.open('dataset/'+random_subject+'/train/good/000.png').resize(imsize).convert('RGB')
        else:
            image_for_cutting = image.copy()
            
        # crop
        left = random.randint(0,imsize[0]-patch_size)
        top = random.randint(0,imsize[1]-patch_size)
        x = image.crop((left,top, left+patch_size, top+patch_size))
        segmentation = segmentation.crop((left,top, left+patch_size, top+patch_size))
        image_for_cutting = transforms.RandomCrop(patch_size)(image_for_cutting)
        
        # coords 
        segmentation = np.array(segmentation.convert('1'))
        coordinates = np.flip(np.column_stack(np.where(segmentation == 1)), axis=1)
        coords = gntr.get_random_coordinate(coordinates)
        
        patch = gntr.generate_patch(
            image_for_cutting,
            area_ratio=CPP.rectangle_area_ratio,
            aspect_ratio=CPP.rectangle_aspect_ratio,
            colorized=True,
            color_type='average'
        )
        patch2 = gntr.generate_patch(
            image_for_cutting,
            area_ratio=CPP.rectangle_area_ratio,
            aspect_ratio=CPP.rectangle_aspect_ratio,
        )
        patch = random.choice([patch, patch2])
        if gntr.check_color_similarity(x, patch) > 0.999:
            low = np.random.uniform(0.5, 0.7)
            high = np.random.uniform(1.3, 1.5)
            patch = ImageEnhance.Brightness(patch).enhance(random.choice([low, high]))
        coords = gntr.check_valid_coordinates_by_container(
            x.size, 
            patch.size, 
            current_coords=coords,
            container_scaling_factor=1
        )
        mask = Image.new('RGB', patch.size, color='black')
        mask2 = gntr.rect2poly(patch, regular=False, sides=8).convert('RGBA')
        mask.paste(mask2, (0,0), None)
        
        
        out = gntr.paste_patch(x, patch, coords, mask2)
        
        if saving_path and not os.path.exists(saving_path):
            os.makedirs(saving_path)
        
        #save data
        plt.figure(figsize=(20,20))
        plt.axis('off')
        plt.imshow(np.array(image))
        plt.savefig(saving_path+'0_good.png', bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(20,20))
        plt.axis('off')
        plt.imshow(np.array(x))
        plt.savefig(saving_path+'1_good_crop.png', bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(20,20))
        plt.axis('off')
        plt.imshow(np.array(segmentation))
        plt.savefig(saving_path+'2_segmentation.png', bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(20,20))
        plt.axis('off')
        plt.imshow(np.array(image_for_cutting))
        plt.savefig(saving_path+'3_image_for_cutting.png', bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(20,20))
        plt.axis('off')
        plt.imshow(np.array(patch))
        plt.savefig(saving_path+'4_defect1.png', bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(20,20))
        plt.axis('off')
        plt.imshow(np.array(mask))
        plt.savefig(saving_path+'5_polygon_mask.png', bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(20,20))
        plt.axis('off')
        plt.imshow(np.array(out))
        plt.savefig(saving_path+'6_final_out_defect1.png', bbox_inches='tight')
        plt.close()
 
 
def show_defect_examples():
    imsize=(256,256)
    patch_size=64
    subject = 'bottle'
    saving_path='brutta_brutta_copia/dataset_analysis/'
    images = f.get_filenames('dataset/'+subject+'/train/good/')
    num_examples = 6
    
    defect1 = []
    defect2 = []
    
    for i in range(num_examples):
        image = Image.open('dataset/'+subject+'/train/good/000.png').resize(imsize).convert('RGB')
        image:Image.Image = transforms.RandomCrop(patch_size)(image)
        patch = gntr.generate_patch(
            image,
            area_ratio=CPP.rectangle_area_ratio,
            aspect_ratio=CPP.rectangle_aspect_ratio,
            colorized=True,
            color_type='average'
        )
        patch2 = gntr.generate_patch(
            image,
            area_ratio=CPP.rectangle_area_ratio,
            aspect_ratio=CPP.rectangle_aspect_ratio,
        )
        patch = random.choice([patch, patch2])
        out = Image.new('RGB', patch.size, color='white')
        mask = gntr.rect2poly(patch, regular=False, sides=8)
        out.paste(patch, (0,0), mask)
        
        
        scar = gntr.generate_patch(
            image,
            area_ratio=CPP.scar_area_ratio,
            aspect_ratio=CPP.scar_aspect_ratio,
        )
        angle = random.randint(-45,45)
        scar = scar.convert('RGBA')
        scar = scar.rotate(angle, expand=True)
        
        defect1.append(out)
        defect2.append(scar)
    
    fig, axs = plt.subplots(1, num_examples, figsize=(20,20))
    for i in range(num_examples):
        axs[i].axis('off')
        axs[i].imshow(defect1[i])
    fig.savefig(saving_path+subject+'/'+'defect1_collection.png', bbox_inches='tight')
    plt.close()
    
    fig, axs = plt.subplots(1, num_examples, figsize=(20,20))
    for i in range(num_examples):
        axs[i].axis('off')
        axs[i].imshow(defect2[i])
    fig.savefig(saving_path+subject+'/'+'defect2_collection.png', bbox_inches='tight')
    plt.close()



def get_polygons():
    image_for_cutting = Image.open('dataset/transistor/train/good/000.png').resize((512,512)).convert('RGB')
    
    fig, axs = plt.subplots(1,9)
    for i in range(9):
        t = np.random.choice([0,1,2], p=[0.5, 0.25, 0.25])
        if t == 0:
            scar = gntr.generate_patch(
                image_for_cutting,
                area_ratio=CPP.rectangle_area_ratio,
                aspect_ratio=CPP.rectangle_aspect_ratio
            )
        if t == 1:
            scar = gntr.generate_patch(
                image_for_cutting,
                area_ratio=CPP.rectangle_area_ratio,
                aspect_ratio=CPP.rectangle_aspect_ratio,
                colorized=True,
                color_type='average'
            )
        if t == 2:
            scar = gntr.generate_patch(
                image_for_cutting,
                area_ratio=CPP.rectangle_area_ratio,
                aspect_ratio=CPP.rectangle_aspect_ratio,
                colorized=True,
                color_type='random'
            )
        mask = gntr.rect2poly(scar, regular=False, sides=8)
        to_paste = Image.new('RGB', mask.size, 'black')
        to_paste.paste(scar, (0,0), mask)
        axs[i].axis('off')
        axs[i].imshow(to_paste)
        
    plt.savefig('polygons.png', bbox_inches='tight')




def get_scar():
    image_for_cutting = Image.open('dataset/cable/train/good/000.png').resize((512,512)).convert('RGB')
    
    fig, axs = plt.subplots(1,9)
    for i in range(9):
        t = np.random.choice([0,1,2], p=[0.5, 0.25, 0.25])
        if t == 0:
            scar = gntr.generate_patch(
                image_for_cutting,
                area_ratio=CPP.scar_area_ratio,
                aspect_ratio=CPP.scar_aspect_ratio
            )
        if t == 1:
            scar = gntr.generate_patch(
                image_for_cutting,
                area_ratio=CPP.scar_area_ratio,
                aspect_ratio=CPP.scar_aspect_ratio,
                colorized=True,
                color_type='average'
            )
        if t == 2:
            scar = gntr.generate_patch(
                image_for_cutting,
                area_ratio=CPP.scar_area_ratio,
                aspect_ratio=CPP.scar_aspect_ratio,
                colorized=True,
                color_type='random'
            )
        angle = random.randint(-45,45)
        s = scar.rotate(angle, expand=True)
        axs[i].axis('off')
        axs[i].imshow(s)
    plt.savefig('scars.png', bbox_inches='tight')

get_polygons()
        
from tqdm import tqdm
from skimage.segmentation import slic
from skimage import color
from self_supervised import constants
from self_supervised.dataset_generator import *
from self_supervised.functional import *
from self_supervised.converters import *
from self_supervised.datasets import PretextTaskDatamodule, CPP
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np
import os
import collections
from skimage.transform import swirl



def do_patch(img, image_for_cutting=None, segmentation=None, patch_loc=False):
    factor = 1 if patch_loc else 1.75
    rectangle_area_ratio = CPP.rectangle_area_ratio_patch if patch_loc else CPP.rectangle_area_ratio 
    k = random.randint(1,2)
    x = img.copy()
    
    segmentation = np.array(segmentation.convert('1'))
    coordinates = np.flip(np.column_stack(np.where(segmentation == 1)), axis=1)

    coords = get_random_coordinate(coordinates)
    t = np.random.choice([0,1,2], p=[0.7, 0.15, 0.15])
    if t == 0:
        patch = generate_patch(
            image_for_cutting,
            area_ratio=rectangle_area_ratio,
            aspect_ratio=CPP.rectangle_aspect_ratio,
        )
    if t == 1:
        patch = generate_patch(
            image_for_cutting,
            area_ratio=rectangle_area_ratio,
            aspect_ratio=CPP.rectangle_aspect_ratio,
            colorized=True,
            color_type='average'
        )
    if t == 2:
        patch = generate_patch(
            image_for_cutting,
            area_ratio=rectangle_area_ratio,
            aspect_ratio=CPP.rectangle_aspect_ratio,
            colorized=True,
            color_type='random'
        )
    #patch = CPP.jitter_transforms(patch)
    if check_color_similarity(img, patch) > 0.99:
        low = np.random.uniform(0.75, 0.9)
        high = np.random.uniform(1.1, 1.15)
        patch = ImageEnhance.Brightness(patch).enhance(random.choice([low, high]))
        patch = ImageEnhance.Brightness(patch).enhance(random.choice([low, high]))
    coords = check_valid_coordinates_by_container(
        img.size, 
        patch.size, 
        current_coords=coords,
        container_scaling_factor=factor
    )
    mask = None
    mask = rect2poly(patch, regular=False, sides=8)
    #mask = mask.filter(ImageFilter.GaussianBlur(1))
    x = paste_patch(x, patch, coords, mask)
    return x


def do_scar(img, image_for_cutting=None, segmentation=None, patch_loc=False):
    factor = 1 if patch_loc else 2
    scar_area_ratio = CPP.scar_area_ratio_patch if patch_loc else CPP.scar_area_ratio
    x = img.copy()
    
    segmentation = np.array(segmentation.convert('1'))
    coordinates = np.flip(np.column_stack(np.where(segmentation == 1)), axis=1)
    
    t = np.random.choice([0,1,2], p=[0.7, 0.15, 0.15])
    if t == 0:
        scar = generate_patch(
            image_for_cutting,
            area_ratio=scar_area_ratio,
            aspect_ratio=CPP.scar_aspect_ratio
        )
    if t == 1:
        scar = generate_patch(
            image_for_cutting,
            area_ratio=scar_area_ratio,
            aspect_ratio=CPP.scar_aspect_ratio,
            colorized=True,
            color_type='average'
        )
    if t == 2:
        scar = generate_patch(
            image_for_cutting,
            area_ratio=scar_area_ratio,
            aspect_ratio=CPP.scar_aspect_ratio,
            colorized=True,
            color_type='random'
        )
    #scar = CPP.jitter_transforms(scar)
    if check_color_similarity(img, scar) > 0.99:
        low = np.random.uniform(0.75, 0.9)
        high = np.random.uniform(1.1, 1.15)
        scar = ImageEnhance.Brightness(scar).enhance(random.choice([low, high]))
        scar = ImageEnhance.Brightness(scar).enhance(random.choice([low, high]))
    scar = scar.convert('RGBA')
    k = random.randint(2,5)
    angle = random.randint(-45,45)
    s = scar.rotate(angle, expand=True)
    for _ in range(k):
        coords = get_random_coordinate(coordinates)
        coords = check_valid_coordinates_by_container(
            img.size, 
            s.size, 
            current_coords=coords,
            container_scaling_factor=factor
        )
        x = paste_patch(x, s, coords, s)
    return x


def do_swirl(img, image_for_cutting=None, segmentation=None, patch_loc=False, subject=None):
    x = img.copy()
    side = random.choice(['left','top'])
    segmentation = np.array(segmentation.convert('1'))
    coords_map = np.flip(np.column_stack(np.where(segmentation == 1)), axis=1)
    
    draw = ImageDraw.Draw(x)
    side = random.choice(['left','top'])
    n = 60 if not patch_loc else 30
    points = []
    c = 0
    for i in range(n):
        offset = i/(n)
        index = random.randint(c, int(len(coords_map)*offset))
        p = coords_map[index]
        points.append(tuple(p))
        c = index

    rgb = random.choice(['black','white','silver'])
    
    if side == 'left':
        points.sort(key=lambda tup: tup[0])
    points = savgol_filter(points, 10, 2, axis=0)
    if not patch_loc:
        p_splits = np.array_split(points, 10)
        k = random.randint(0,9)
        points = p_splits[k]
    #else:
    #    p_splits = np.array_split(points, 3)
    #    k = random.randint(0,2)
    #    points = p_splits[k]
    
    points = [tuple(x) for x in points]
    if patch_loc:
        draw.line(
            points, fill=rgb, width=1)
    else:
        draw.line(
            points, fill=rgb, width=3)
    return x
    

def plot_together(good, def1, def2=None, def3=None, masks=None, saving_path=None, name='plot.png'):
    if saving_path and not os.path.exists(saving_path):
        os.makedirs(saving_path)
    img = good[0]
    shape = img.shape[0]
    hseparator = Image.new(mode='RGB', size=(6,shape), color=(255,255,255))
    
    good_images = np.hstack([np.hstack(
      [np.array(good[i]), np.array(hseparator)]
      ) if i < len(good)-1 else np.array(good[i]) for i in range(len(good))])   
    
    if def1 is not None:
        def1 = np.hstack([np.hstack(
        [np.array(def1[i]), np.array(hseparator)]
        ) if i < len(def1)-1 else np.array(def1[i]) for i in range(len(def1))])
    
    if def2 is not None:
        def2 = np.hstack([np.hstack(
        [np.array(def2[i]), np.array(hseparator)]
        ) if i < len(def2)-1 else np.array(def2[i]) for i in range(len(def2))])
    
    if def3 is not None:
        def3 = np.hstack([np.hstack(
        [np.array(def3[i]), np.array(hseparator)]
        ) if i < len(def3)-1 else np.array(def3[i]) for i in range(len(def3))])
    
    masks = np.hstack([np.hstack(
      [np.array(masks[i]), np.array(hseparator)]
      ) if i < len(masks)-1 else np.array(masks[i]) for i in range(len(masks))])
    
    vseparator = np.array(Image.new(
      mode='RGB', 
      size=(good_images.shape[1], 6), 
      color=(255,255,255)))
    
    tot = np.vstack([good_images, vseparator])
    
    if def1 is not None:
        tot = np.vstack([tot, def1, vseparator])
    
    if def2 is not None:
        tot = np.vstack([tot, def2, vseparator])
    
    if def3 is not None:
        tot = np.vstack([tot, def3, vseparator])
    
    tot = np.vstack([tot, masks])
        
        
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
    temp_imsize = (256,256)
    patchsize = 32
    subjects = get_all_subject_experiments('dataset/')
    classes = get_all_subject_experiments('dataset/')
    subjects = ['cable','metal_nut']
    for i in tqdm(range(len(subjects))):
        sub = subjects[i]
        images = get_filenames('dataset/'+sub+'/train/good/')
        goods = []
        masks = []
        patches = []
        scars = []
        swirls = []
        
        img = Image.open('dataset/'+sub+'/train/good/000.png').resize(temp_imsize).convert('RGB')
        
        if sub in np.array(['carpet','grid','leather','tile','wood']):
            fixed_mask = Image.new(size=temp_imsize, mode='RGB', color='white')
        else:
            to_mask = img.copy()
            if sub == 'cable':
                image_array = np.array(to_mask)
                segments = slic(image_array, n_segments = 5, sigma =2, convert2lab=True)
                superpixels = color.label2rgb(segments, image_array, kind='avg')
                to_mask = Image.fromarray(superpixels).convert('RGB')
            fixed_mask = obj_mask(to_mask)
        
        
        for i in tqdm(range(12)):
            img = Image.open(images[i]).resize(temp_imsize).convert('RGB')
            # create new masks only for non-fixed objects
            if sub in constants.NON_FIXED_OBJECTS():
                mask = obj_mask(img)
            else:
                mask = fixed_mask
            
            if sub in np.array(['carpet','grid','leather','tile','wood']):
                random_subject = random.choice(classes)
                cutting = Image.open('dataset/'+random_subject+'/train/good/000.png').resize(temp_imsize).convert('RGB')
            else:
                cutting = img.copy()
            # crop if patch localization
            if patch_localization:
                # preventive crop for capsule and screw
                if sub == 'capsule':
                    img = img.crop((0,50,255,200))
                    mask = mask.crop((0,50,255,200))
                if sub == 'screw':
                    img = img.crop((25,25,230,230))
                    mask = mask.crop((25,25,230,230))
                left = random.randint(0,img.size[0]-patchsize)
                top = random.randint(0,img.size[1]-patchsize)
                x = img.crop((left,top, left+patchsize, top+patchsize))
                mask = mask.crop((left,top, left+patchsize, top+patchsize))
                cutting = transforms.RandomCrop(patchsize)(cutting)
            else:
                affine = transforms.RandomAffine(3, scale=(1.05,1.1))
                img = affine(img)
                x = img.copy()
            if torch.sum(transforms.ToTensor()(mask)) > int((patchsize*patchsize)/2):
                patch = do_patch(x,image_for_cutting=cutting, segmentation=mask, patch_loc=patch_localization)
                scar = do_scar(x,image_for_cutting=cutting, segmentation=mask, patch_loc=patch_localization)
                swirl = do_swirl(x,image_for_cutting=cutting, segmentation=mask, patch_loc=patch_localization, subject=sub)
            else:
                patch = x.copy()
                scar = x.copy()
                swirl = x.copy()
            
            x = x.resize(imsize)
            mask = mask.resize(imsize)
            patch = patch.resize(imsize)
            scar = scar.resize(imsize)
            swirl = swirl.resize(imsize)
            x = CPP.jitter_transforms(x)
            patch = CPP.jitter_transforms(patch)
            scar = CPP.jitter_transforms(scar)
            swirl = CPP.jitter_transforms(swirl)
            goods.append(np.array(x))
            patches.append(np.array(patch))
            scars.append(np.array(scar))
            masks.append(np.array(mask))
            swirls.append(np.array(swirl))
        
        if patch_localization:
            plot_together(goods, patches, scars, swirls, masks, 'brutta_copia/a/patch_level/dataset_analysis/'+sub+'/', sub+'_artificial_crop.png')
        else:
            plot_together(goods, patches, scars, swirls, masks, 'brutta_copia/a/image_level/dataset_analysis/'+sub+'/', sub+'_artificial.png')
        os.system('clear')
        
        
def check_all_subject(patch_localization = False):
    subjects = get_all_subject_experiments('dataset/')
    imsize=(256,256)
    temp_imsize = (256,256)
    patchsize = 32
    goods = []
    masks = []
    patches = []
    scars = []
    swirls = []
    classes = get_all_subject_experiments('dataset/')
    
    
    for i in tqdm(range(len(subjects))):
        subject = subjects[i]
        img = Image.open('dataset/'+subject+'/train/good/000.png').resize(temp_imsize).convert('RGB') 
        
        
        # mask generation    
        if subject in np.array(['carpet','grid','leather','tile','wood']):
            mask = Image.new(size=temp_imsize, mode='RGB', color='white')
        else:
            to_mask = img.copy()
            if subject == 'cable':
                image_array = np.array(to_mask)
                segments = slic(image_array, n_segments=5, sigma=2, convert2lab=True)
                superpixels = color.label2rgb(segments, image_array, kind='avg')
                to_mask = Image.fromarray(superpixels).convert('RGB')
            mask = obj_mask(to_mask)

        # image for cut
        if subject in np.array(['carpet','grid','leather','tile','wood']):
            random_subject = random.choice(classes)
            cutting = Image.open('dataset/'+random_subject+'/train/good/000.png').resize(temp_imsize).convert('RGB')
        else:
            cutting = img.copy()
        # patch loc -> crop
        if patch_localization:
            if subject == 'capsule':
                img = img.crop((0,50,255,200))
                mask = mask.crop((0,50,255,200))
            if subject == 'screw':
                img = img.crop((25,25,230,230))
                mask = mask.crop((25,25,230,230))
            left = random.randint(0,img.size[0]-patchsize)
            top = random.randint(0,img.size[1]-patchsize)
            x = img.crop((left,top, left+patchsize, top+patchsize))
            mask = mask.crop((left,top, left+patchsize, top+patchsize))
            cutting = transforms.RandomCrop(patchsize)(img)
        else:
            if subject not in constants.NON_FIXED_OBJECTS():
                affine = transforms.RandomAffine(3, scale=(1.05,1.1))
                img = affine(img)
            x = img.copy()
        if torch.sum(transforms.ToTensor()(mask)) > int((patchsize*patchsize)/2):
            patch = do_patch(x,image_for_cutting=cutting, segmentation=mask, patch_loc=patch_localization)
            scar = do_scar(x,image_for_cutting=cutting, segmentation=mask, patch_loc=patch_localization)
            swirl = do_swirl(x,image_for_cutting=cutting, segmentation=mask, patch_loc=patch_localization, subject=subject)
        else:
            patch = x.copy()
            scar = x.copy()
            swirl = x.copy()
        
        #x = x.resize(imsize)
        #mask = mask.resize(imsize)
        #patch = patch.resize(imsize)
        #scar = scar.resize(imsize)
        #swirl = swirl.resize(imsize)
        x = CPP.jitter_transforms(x)
        patch = CPP.jitter_transforms(patch)
        scar = CPP.jitter_transforms(scar)
        swirl = CPP.jitter_transforms(swirl)
        goods.append(np.array(x))
        patches.append(np.array(patch))
        scars.append(np.array(scar))
        masks.append(np.array(mask))
        swirls.append(np.array(swirl))
    
    if patch_localization:
        plot_together(goods, patches, scars, swirls, masks, 'brutta_copia/a/patch_level/dataset_analysis/', 'artificial_overall_crop.png')
    else:
        plot_together(goods, patches, scars, swirls, masks, 'brutta_copia/a/image_level/dataset_analysis/', 'artificial_overall.png')

    os.system('clear')


def check_distribution():
    artificial = PretextTaskDatamodule(
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



def fixed_good_bad():
    defects_filenames = [
        'dataset/tile/test/oil/000.png',
        'dataset/tile/test/gray_stroke/000.png',
        'dataset/wood/test/liquid/000.png',
        'dataset/wood/test/color/001.png',
        'dataset/grid/test/metal_contamination/005.png',
        'dataset/grid/test/thread/005.png'
    ]
    ground_truths_filenames = [
        'dataset/tile/ground_truth/oil/000_mask.png',
        'dataset/tile/ground_truth/gray_stroke/000_mask.png',
        'dataset/wood/ground_truth/liquid/000_mask.png',
        'dataset/wood/ground_truth/color/001_mask.png',
        'dataset/grid/ground_truth/metal_contamination/005_mask.png',
        'dataset/grid/ground_truth/thread/005_mask.png'
    ]
    defects = []
    ground_truths = []
    for i in range(len(defects_filenames)):
        defect = Image.open(defects_filenames[i]).resize((256,256)).convert('RGB')
        ground_truth = get_ground_truth(ground_truths_filenames[i]).convert('RGB')
        defects.append(np.array(defect))
        ground_truths.append(np.array(ground_truth))
    defects = np.array(defects)
    ground_truths = np.array(ground_truths)
    
    plot_together(defects, None, None, None, ground_truths, 'brutta_copia/bho/dataset_analysis/', 'contaminations.png')
        
def good_and_bad():
    subjects = ['tile', 'woods', 'grid']
    
    goods = []
    defects = []
    ground_truths = []
    for sub in subjects:
        image = Image.open('dataset/'+sub+'/train/good/000.png').resize((256,256)).convert('RGB')
        d = get_subdirectories('dataset/'+sub+'/test/')
        defect = Image.open('dataset/'+sub+'/test/'+d[1]+'/000.png').resize((256,256)).convert('RGB')
        ground_truth = get_ground_truth('dataset/'+sub+'/ground_truth/'+d[1]+'/000_mask.png').convert('RGB')
        
        goods.append(np.array(image))
        defects.append(np.array(defect))
        ground_truths.append(np.array(ground_truth))
        
        image = Image.open('dataset/'+sub+'/train/good/001.png').resize((256,256)).convert('RGB')
        d = get_subdirectories('dataset/'+sub+'/test/')
        defect = Image.open('dataset/'+sub+'/test/'+d[1]+'/001.png').resize((256,256)).convert('RGB')
        ground_truth = get_ground_truth('dataset/'+sub+'/ground_truth/'+d[1]+'/001_mask.png').convert('RGB')

        goods.append(np.array(image))
        defects.append(np.array(defect))
        ground_truths.append(np.array(ground_truth))
        
    goods = np.array(goods)
    defects = np.array(defects)
    ground_truths = np.array(ground_truths)
    
    plot_together(goods, defects, None, None, ground_truths, 'brutta_copia/bho/dataset_analysis/', 'originals.png')

if __name__ == "__main__":
    #fixed_good_bad()
    #good_and_bad()
    test_augmentations(True)
    check_all_subject(True)
    #check_distribution()
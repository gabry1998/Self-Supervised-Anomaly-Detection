from self_supervised.support.dataset_generator import *
from self_supervised.support.functional import *
from self_supervised.support.cutpaste_parameters import CPP
from skimage import feature
from skimage.morphology import square, label
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time


def do_patch(img):
    start = time.time()
    segmentation = obj_mask(img)
    coords = get_random_coordinate(segmentation)
    patch = generate_patch(img, augs=CPP.jitter_transforms)
    coords, center = get_coordinates_by_container(
        img.size, 
        patch.size, 
        current_coords=coords,
        container_scaling_factor=1.75)
    mask = None
    mask = polygonize(patch, 3,9)
    out = paste_patch(img, patch, coords, mask, center=center, debug=True)
    end = time.time() - start
    print('patch created in', end, 'sec')
    return out


def do_scar(img):
    start = time.time()
    segmentation = obj_mask(img)
    coords = get_random_coordinate(segmentation)
    scar= generate_scar(
            img,
            colorized=True,
            with_padding=True,
            augs=CPP.jitter_transforms
        )
    mask = scar.copy()
    #mask = polygonize(scar, 3, 9)
    angle = random.randint(-45,45)
    mask = mask.rotate(angle)
    scar = scar.rotate(angle)
    coords, center = get_coordinates_by_container(
        img.size, 
        scar.size, 
        current_coords=coords,
        container_scaling_factor=2.5)
    out = paste_patch(img, scar, coords, mask, center=center, debug=True)
    end = time.time() - start
    print('scar created in', end, 'sec')
    return out


def largest_cc(segmentation):
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC

def obj_mask(image):
    gray = np.array(image.convert('L'))
    edged_image = feature.canny(gray, sigma=1.5, low_threshold=5, high_threshold=15)
    structure = square(3)
    edged_image = binary_dilation(edged_image, structure)
    edged_image = binary_closing(edged_image, structure)
    edged_image = ndimage.binary_fill_holes(edged_image, structure)
    structure = square(4)
    edged_image = binary_erosion(edged_image, structure)
    edged_image = (edged_image*255).astype(np.uint8)
    edged_image = largest_cc(edged_image)
    return edged_image


def get_random_coordinate(binary_mask):
    xy_coords = np.flip(np.column_stack(np.where(binary_mask == 1)), axis=1)
    idx = random.randint(0, len(xy_coords)-1)
    return xy_coords[idx]

def do_mask(image):
    start = time.time()
    mask = obj_mask(image)
    end = time.time() - start
    print('mask created in', end, 'sec')
    return Image.fromarray(mask).convert('RGB')

def do_swirl(img):
    start = time.time()
    coords, _ = get_coordinates_by_container(img.size, (0,0), 2.5)
    scar = generate_swirl(
            img,
            coords,
            swirl_strength=(2,5),
            swirl_radius=(50,100)
        )
    end = time.time() - start
    print('swirl created in', end, 'sec')
    return scar


def save_fig(my_array, name):
    hseparator = Image.new(mode='RGB', size=(6,256), color=(255,255,255))
    my_array = np.hstack([np.hstack(
      [np.array(my_array[i]), np.array(hseparator)]
      ) if i < len(my_array)-1 else np.array(my_array[i]) for i in range(len(my_array))])
    plt.figure(figsize=(30,30))
    plt.imshow(my_array)
    plt.axis('off')
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def test_augmentations():
    subjects = get_image_filenames('dataset/screw/train/good/')
    imsize=(256,256)
    masks = []
    patches = []
    scars = []
    for i in range(10):
        img = Image.open(subjects[i]).resize(imsize).convert('RGB')
        patch = do_patch(img)
        scar = do_scar(img)
        mask = do_mask(img)
        patches.append(np.array(patch))
        scars.append(np.array(scar))
        masks.append(mask)
    
    save_fig(patches, 'patch.png')
    save_fig(scars, 'scar.png')
    save_fig(masks, 'mask.png')
     

def check_all_subject():
    subjects = get_all_subject_experiments('dataset/')
    imsize=(256,256)
    masks = []
    patches = []
    scars = []
    for subject in subjects:
        img = Image.open('dataset/'+subject+'/train/good/000.png').resize(imsize).convert('RGB')
        patch = do_patch(img)
        scar = do_scar(img)
        mask = do_mask(img)
        masks.append(np.array(mask))
        patches.append(np.array(patch))
        scars.append(np.array(scar))
    
    save_fig(patches, 'patch.png')
    save_fig(scars, 'scar.png')
    save_fig(masks, 'mask.png')
    
    
test_augmentations()
#check_all_subject()



import random
from PIL import Image,ImageDraw, ImageFilter
import numpy as np
from .functional import normalize_in_interval
from scipy.spatial import ConvexHull
from skimage.transform import swirl
from skimage.morphology import square, label
from skimage import feature
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing



class Container:
    def __init__(self, imsize:tuple, scaling_factor:float) -> None:
        self.center = int(imsize[0]/2)
        self.dim = int(imsize[0]/scaling_factor)
        self.left = int(self.center-(self.center/scaling_factor))
        self.top = int(self.center-(self.center/scaling_factor))
        self.right = int(self.center+(self.center/scaling_factor))
        self.bottom = int(self.center+(self.center/scaling_factor))
        self.width = self.right - self.left
        self.height = self.bottom - self.top


def obj_mask(image, patch_localization:bool=False):
    if patch_localization:
        image = image.filter(ImageFilter.SHARPEN)
    gray = np.array(image.convert('L'))
    if patch_localization:
        edged_image = feature.canny(gray,sigma=1.5, high_threshold=30)
    else:
        edged_image = feature.canny(gray, sigma=1.5, low_threshold=5, high_threshold=15)
        structure = square(3)
        edged_image = binary_dilation(edged_image, structure).astype(int)
        edged_image = binary_closing(edged_image, structure)
        edged_image = ndimage.binary_fill_holes(edged_image, structure).astype(int)
        structure = square(4)
        edged_image = binary_erosion(edged_image, structure).astype(int)
    edged_image = (edged_image*255).astype(np.uint8)
    labels = label(edged_image)
    edged_image = labels == np.argmax(np.bincount(labels.flat, weights=edged_image.flat))
    return edged_image


def polygonize(patch, min_points:int=5, max_points:int=15):
    mask = Image.new('RGBA', (patch.size), (0,0,0,0)) 
    draw = ImageDraw.Draw(mask)
    
    points = get_random_points(
        mask.size[0],
        mask.size[1],
        min_points,
        max_points)
    draw.polygon(points, fill='white')
    return mask


def get_coordinates_by_container(
        imsize:tuple, 
        patchsize:tuple,
        current_coords:tuple=None,
        container_scaling_factor:int=1):
    patch_w, patch_h = patchsize
    container = Container(imsize, scaling_factor=container_scaling_factor)
    # coordinate
    if current_coords is None:
        center_x = random.randint(container.left, container.right)
        center_y = random.randint(container.top, container.bottom)
    else:
        center_x = current_coords[0]
        center_y = current_coords[1]

    paste_left = center_x - int(patchsize[0]/2)
    paste_top = center_y - int(patchsize[1]/2)
    
    if paste_left < container.left:
        center_x = container.left
        paste_left = center_x - int(patchsize[0]/2)
        if paste_left < 0:
            paste_left = 0
            center_x = int(patch_w/2)
    if paste_left > container.right:
        center_x = container.right
        paste_left = center_x - int(patchsize[0]/2)
    if paste_top < container.top: 
        center_y = container.top
        paste_top = center_y - int(patchsize[1]/2)
        if paste_top < 0:
            paste_top = 0
            center_y = int(patch_h/2)
    if paste_top > container.bottom:
        center_y = container.bottom
        paste_top = center_y - int(patchsize[1]/2)
    return (paste_left, paste_top), (center_x, center_y)
   
    
def generate_patch(
        image, 
        area_ratio:tuple=(0.02, 0.15), 
        aspect_ratio:tuple=((0.3, 1),(1, 3.3)),
        augs=None):

    img_area = image.size[0] * image.size[1]
    patch_area = random.uniform(area_ratio[0], area_ratio[1]) * img_area
    patch_aspect = random.choice([random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])])
    patch_w  = int(np.sqrt(patch_area*patch_aspect))
    patch_h = int(np.sqrt(patch_area/patch_aspect))
    org_w, org_h = image.size

    # parte da tagliare
    patch_left, patch_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
    patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h
        
    cropped_patch = image.crop((patch_left, patch_top, patch_right, patch_bottom))
    if augs:
        cropped_patch = augs(cropped_patch)
    return cropped_patch


def generate_swirl(
        image,
        center:tuple,
        swirl_strength:tuple=(2,5),
        swirl_radius:tuple=(50,100)):
    img_arr = np.array(image)
    
    if swirl_radius[1] > image.size[0]:
        swirl_radius = (int(image.size[0]/4), int(image.size[0]/2))
    r = random.randint(swirl_radius[0], swirl_radius[1])

    warped = swirl(
        img_arr, 
        center=center,
        rotation=0, 
        strength=random.randint(
            swirl_strength[0],
            swirl_strength[1]), 
        radius=r)
    warped = np.array(warped*255, dtype=np.uint8)
    warped = Image.fromarray(warped, image.mode)
    return warped


def generate_scar(
        image, 
        w_range:tuple=(2,16), 
        h_range:tuple=(10,25),  
        with_padding:bool=False,
        colorized:bool=False,
        augs=None):
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
    
    
    if colorized:
        r = random.randint(30,220)
        g = random.randint(30,220)
        b = random.randint(30,220)
        color = (r,g,b)
        scar = Image.new('RGBA', (scar_w, scar_h), color=color)
    else:
        scar = image.crop((patch_left, patch_top, patch_right, patch_bottom))
        if augs:
            scar = augs(scar)
        if with_padding:
            padding = Image.new(image.mode, (new_width, new_height), (255, 255, 255))
            padding.paste(scar, (left, top))
            scar = padding
        scar = scar.convert('RGBA')
    return scar


def get_random_coordinate(binary_mask):
    xy_coords = np.flip(np.column_stack(np.where(binary_mask == 1)), axis=1)
    idx = random.randint(0, len(xy_coords)-1)
    return xy_coords[idx]


def get_random_points(width, height,min_num_points=3, max_num_points=4):
    raw_points = 0.1 + 0.8*np.random.rand(random.randint(min_num_points,max_num_points), 2)
    ch = ConvexHull(raw_points)
    hull_indices = ch.vertices
    points = raw_points[hull_indices, :]
    x = [points[i][0] for i in range(len(points))]
    y = [points[i][1] for i in range(len(points))]
    x1 = normalize_in_interval(x, 0, width)
    y1 = normalize_in_interval(y, 0, height)
    return [(x1[i], y1[i]) for i in range(len(points))]


def paste_patch(image, patch, coords, mask=None, center:tuple=None, debug:bool=False):
    aug_image = image.copy()
    aug_image.paste(patch, (coords[0], coords[1]), mask=mask)
    if debug:
        aug_image.paste(Image.new('RGB', (2,2), 'red'), (coords[0], coords[1]), None)
        aug_image.paste(Image.new('RGB', (2,2), 'green'), (center[0], center[1]), None)
    return aug_image


def random_color():
    return random.randint(10,240)


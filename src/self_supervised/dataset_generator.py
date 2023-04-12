from PIL import Image,ImageDraw
from skimage.morphology import square, label
from sklearn.metrics.pairwise import cosine_similarity
from skimage import feature
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing
from torchvision.transforms import ColorJitter
from array import ArrayType
from typing import Tuple
import numpy as np
import random



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


def obj_mask(image:Image.Image) -> Image.Image:
    gray = np.array(image.convert('L'))
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
    return Image.fromarray(edged_image).convert('RGB')


def rect2poly(patch:Image.Image, regular:bool=False, sides:list=4):
    width, height = patch.size
    mask = Image.new('RGBA', (patch.size), color=(0,0,0,0))
    draw = ImageDraw.Draw(mask)
    if regular:
        max_val = int(min([width, height])/2)
        cx = int(width/2)
        cy = int(height/2)
        draw.regular_polygon(
            bounding_circle=((cx,cy),max_val),
            n_sides=random.choice(sides),
            fill='white'
        )
    else:
        if sides == 4:
            points = [
                (0, random.randint(1, height)), #left
                (random.randint(1, width), 0), #top
                (width, random.randint(1, height)), #right
                (random.randint(1, width), height), #bottom
                ]
        else:
            points = []
            for side in range(4):
                num_points_per_side = random.randint(1,2)
                if side == 0: # left
                    if num_points_per_side == 1:
                        points.append( (0, random.randint(1, height)) )
                    else:
                        p1 = (0, random.randint(int(height/2)+1, height))
                        p2 = (0, random.randint(1, int(height/2)))
                        points.append(p1)
                        points.append(p2)
                if side == 1: # top
                    if num_points_per_side == 1:
                        points.append( (random.randint(1, width), 0) )
                    else:
                        p1 = (random.randint(1, int(width/2)), 0)
                        p2 = (random.randint(int(width/2)+1, width), 0)
                        points.append(p1)
                        points.append(p2)
                if side == 2: # right
                    if num_points_per_side == 1:
                        points.append( (width, random.randint(1, height)) )
                    else:
                        p1 = (width, random.randint(1, int(height/2)))
                        p2 = (width, random.randint(int(height/2)+1, height))
                        points.append(p1)
                        points.append(p2)
                if side == 3: # bottom
                    if num_points_per_side == 1:
                        points.append( (random.randint(1, width), height) )
                    else:
                        p1 = (random.randint(int(width/2)+1, width), height)
                        p2 = (random.randint(1, int(width/2)), height)
                        points.append(p1)
                        points.append(p2)
        draw.polygon(
            points, fill='white')
    return mask


def check_valid_coordinates_by_container(
        imsize:tuple, 
        patchsize:tuple,
        current_coords:tuple=None,
        container_scaling_factor:int=1):
    defect_width, defect_height = patchsize
    container = Container(imsize, scaling_factor=container_scaling_factor)
    # no coords? generate brand new
    if current_coords is None:
        center_x = random.randint(container.left, container.right)
        center_y = random.randint(container.top, container.bottom)
    else:
        center_x = current_coords[0]
        center_y = current_coords[1]

    # we have (x, y); PIL paste() requires (left, top)
    paste_left:int = center_x - int(patchsize[0]/2)
    paste_top:int = center_y - int(patchsize[1]/2)
    paste_right:int = center_x + int(patchsize[0]/2)
    paste_bottom:int = center_y + int(patchsize[1]/2)
    
    # check defect bounding box (left, top, right, bottom) is in container, and update (left, top)
    
    #right point
    if paste_right > container.right:
        paste_left:int = container.right - defect_width
        center_x:int = paste_right - int(defect_width/2)
    #bottom point
    if paste_bottom > container.bottom:
        paste_top:int = container.bottom - defect_height
        center_y:int = paste_bottom - int(defect_height/2)
    # left point
    if paste_left < container.left:
        paste_left:int = container.left
        center_x:int = container.left + int(defect_width/2)
    # top point
    if paste_top < container.top:
        paste_top:int = container.top
        center_y:int = container.top + int(defect_height/2)
        
    return (paste_left, paste_top)


def check_color_similarity(patch:Image.Image, defect:Image.Image) -> float:
    imarray = np.array(patch)
    color = imarray.mean(axis=(0,1))
    rgb1 = (float(color[0]/255), float(color[1]/255), float(color[2]/255))
    
    imarray = np.array(defect)
    color = imarray.mean(axis=(0,1))
    rgb2 = (float(color[0]/255), float(color[1]/255), float(color[2]/255))
    
    v1 = np.array(rgb1)[None, :]
    v2 = np.array(rgb2)[None, :]
    out = cosine_similarity(v1, v2)
    return out.squeeze()



   
def generate_patch(
        image:Image.Image, 
        area_ratio:tuple=(0.02, 0.15), 
        aspect_ratio:tuple=((0.3, 1),(1, 3.3)),
        augs:ColorJitter=None,
        colorized:bool=False,
        color_type:str='random') -> Image.Image:

    img_area = image.size[0] * image.size[1]
    patch_area = random.uniform(area_ratio[0], area_ratio[1]) * img_area
    patch_aspect = random.choice([random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])])
    patch_w  = int(np.sqrt(patch_area*patch_aspect))
    patch_h = int(np.sqrt(patch_area/patch_aspect))
    if patch_w < 2:
        patch_w = 2
    if patch_h < 2:
        patch_h = 2
    org_w, org_h = image.size

    # parte da tagliare
    w = org_w - patch_w
    if w < 1:
        w = 1
    h = org_h - patch_h
    if h < 1:
        h = 1
    patch_left, patch_top = random.randint(0, w), random.randint(0, h)
    patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h
    
    if colorized:
        if color_type=='random':
            rgb = (
                random.randint(0,255),
                random.randint(0,255),
                random.randint(0,255)
            )
        elif color_type=='sample':
            rgb = random.choice(['black','white','silver', 'gray'])
        elif color_type=='average':
            patch = image.crop((patch_left, patch_top, patch_right, patch_bottom))
            imarray = np.array(patch)
            color = imarray.mean(axis=(0,1))
            rgb = (int(color[0]), int(color[1]), int(color[2]))
        cropped_patch = Image.new('RGB', (patch_w, patch_h), color=rgb)
    else:
        cropped_patch = image.crop((patch_left, patch_top, patch_right, patch_bottom))
    return cropped_patch


def generate_scar(
        image:Image.Image, 
        w_range:tuple=(2,16), 
        h_range:tuple=(10,25),  
        with_padding:bool=False,
        colorized:bool=False,
        augs:ColorJitter=None,
        color_type:str='random') -> Image.Image:
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
        if color_type=='random':
            rgb = (
                random.randint(30,225),
                random.randint(30,225),
                random.randint(30,225)
            )
        elif color_type=='sample':
            rgb = random.choice(['green','red','yellow','blue','orange','cyan','purple'])
        elif color_type=='average':
            scar = image.crop((patch_left, patch_top, patch_right, patch_bottom))
            imarray = np.array(scar)
            color = imarray.mean(axis=(0,1))
            rgb = (int(color[0]), int(color[1]), int(color[2]))
        scar = Image.new('RGBA', (scar_w, scar_h), color=rgb)
    else:
        scar = image.crop((patch_left, patch_top, patch_right, patch_bottom))
        if with_padding:
            padding = Image.new(image.mode, (new_width, new_height), color='silver')
            padding.paste(scar, (left, top))
            scar = padding
        scar = scar.convert('RGBA')
    return scar


def get_random_coordinate(xy_coords:ArrayType) -> Tuple[int, int]:
    if len(xy_coords) == 0:
        return None
    elif len(xy_coords) < 2:
        return xy_coords[0]
    idx = random.randint(0, len(xy_coords)-1)
    return xy_coords[idx]


def paste_patch(
        image:Image.Image, 
        patch:Image.Image, 
        coords:tuple, 
        mask:Image.Image=None):
    aug_image = image.copy()
    aug_image.paste(patch, (coords[0], coords[1]), mask=mask)
    return aug_image


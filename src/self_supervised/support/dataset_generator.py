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
import cv2



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


def obj_mask(image):
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


def rect2poly(patch, regular:bool=False, sides:list=4):
    width, height = patch.size
    mask = Image.new('RGBA', (patch.size), (0,0,0,0)) 
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
        augs=None,
        colorized:bool=False,
        color_type:str='random'):

    img_area = image.size[0] * image.size[1]
    patch_area = random.uniform(area_ratio[0], area_ratio[1]) * img_area
    patch_aspect = random.choice([random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])])
    patch_w  = int(np.sqrt(patch_area*patch_aspect))
    patch_h = int(np.sqrt(patch_area/patch_aspect))
    org_w, org_h = image.size

    # parte da tagliare
    patch_left, patch_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
    patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h
    
    if colorized:
        if color_type=='random':
            rgb = (
                random.randint(0,255),
                random.randint(0,255),
                random.randint(0,255)
            )
        elif color_type=='sample':
            rgb = random.choice(['black','white','green','red','yellow','orange','cyan'])
        elif color_type=='average':
            patch = image.crop((patch_left, patch_top, patch_right, patch_bottom))
            imarray = np.array(patch)
            color = imarray.mean(axis=(0,1))
            rgb = (int(color[0]), int(color[1]), int(color[2]))
        cropped_patch = Image.new('RGBA', (patch_w, patch_h), color=rgb)
    else:
        cropped_patch = image.crop((patch_left, patch_top, patch_right, patch_bottom))

    if augs and colorized==False:
        cropped_patch = augs(cropped_patch)
    return cropped_patch


def generate_scar(
        image, 
        w_range:tuple=(2,16), 
        h_range:tuple=(10,25),  
        with_padding:bool=False,
        colorized:bool=False,
        augs=None,
        color_type='random'):
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
        if augs:
            scar = augs(scar)
        if with_padding:
            padding = Image.new(image.mode, (new_width, new_height), color='silver')
            padding.paste(scar, (left, top))
            scar = padding
        scar = scar.convert('RGBA')
    return scar


def get_random_coordinate(binary_mask):
    binary_mask = np.array(binary_mask.convert('1'))
    xy_coords = np.flip(np.column_stack(np.where(binary_mask == 1)), axis=1)
    if len(xy_coords) == 0:
        return None
    elif len(xy_coords) < 2:
        return xy_coords[0]
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


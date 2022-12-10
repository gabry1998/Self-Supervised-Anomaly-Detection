import random
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import numpy as np
from .functional import normalize_in_interval
from scipy.spatial import ConvexHull



class Deformer:
    def __init__(self, imsize:tuple, points:tuple) -> None:
        self.top, self.left, self.bottom, self.right = imsize
        self.crop_left, self.crop_top,self.crop_right, self.crop_bottom = points
        
    def getmesh(self, img):
        return [(
                # target rectangle
                (self.crop_left, self.crop_top,self.crop_right, self.crop_bottom),
                # corresponding source quadrilateral
                (np.random.randint(self.top, self.bottom), 
                 np.random.randint(self.left, self.right),
                 np.random.randint(self.top, self.bottom),
                 np.random.randint(self.left, self.right),
                 np.random.randint(self.top, self.bottom),
                 np.random.randint(self.left, self.right),
                 np.random.randint(self.top, self.bottom),
                 np.random.randint(self.left, self.right))
                )]


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


def generate_patch_new(
        image, 
        area_ratio:tuple=(0.02, 0.15), 
        aspect_ratio:tuple=((0.3, 1),(1, 3.3)),
        polygoned=False,
        distortion=False,
        factor=1.75):

    img_area = image.size[0] * image.size[1]
    patch_area = random.uniform(area_ratio[0], area_ratio[1]) * img_area
    patch_aspect = random.choice([random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])])
    patch_w  = int(np.sqrt(patch_area*patch_aspect))
    patch_h = int(np.sqrt(patch_area/patch_aspect))
    org_w, org_h = image.size
    container = Container(image.size, scaling_factor=factor)

    # parte da tagliare
    patch_left, patch_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
    patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h

    
    # coordinate
    if (container.right-patch_w) > container.left:
        paste_left = random.randint(container.left, container.right-patch_w)
    else:
        paste_left = container.left
    if (container.bottom-patch_h) > container.top:
        paste_top = random.randint(container.top, container.bottom-patch_h)
    else:
        paste_top = container.top
    mask = None
    
    if polygoned:
        mask = Image.new('RGBA', (patch_w, patch_h), (255,255,255,0)) 
        draw = ImageDraw.Draw(mask)
        
        points = get_random_points(
            mask.size[0],
            mask.size[1],
            5,
            15)
        draw.polygon(points, fill='black')
        
    if distortion:
        deformer = Deformer(
            (0, 0, image.size[0], image.size[1]), 
            points=(patch_left, patch_top, patch_right, patch_bottom))
        deformed_image = ImageOps.deform(image, deformer)
        cropped_patch = deformed_image.crop((patch_left, patch_top, patch_right, patch_bottom))
    else:
        cropped_patch = image.crop((patch_left, patch_top, patch_right, patch_bottom))
    return cropped_patch, mask, (paste_left, paste_top)


def generate_scar_centered(
        image, 
        w_range:tuple=(2,16), 
        h_range:tuple=(10,25), 
        augs=None, 
        with_padding:bool=False,
        colorized:bool=False,
        factor=2.5):
    img_w, img_h = image.size
    right = 1
    left = 1
    top = 1
    bottom = 1
    container = Container(image.size, scaling_factor=factor)
    scar_w = random.randint(w_range[0], w_range[1])
    scar_h = random.randint(h_range[0], h_range[1])
    new_width = scar_w + right + left
    new_height = scar_h + top + bottom
    patch_left, patch_top = random.randint(0, img_w - scar_w), random.randint(0, img_h - scar_h)
    patch_right, patch_bottom = patch_left + scar_w, patch_top + scar_h
    
    
    if colorized:
        r = random_color()
        g = random_color()
        b = random_color()
        color = (r,g,b)
        scar = Image.new('RGBA', (scar_w, scar_h), color=color)
    else:
        scar = image.crop((patch_left, patch_top, patch_right, patch_bottom))
        if with_padding:
            scar_with_pad = Image.new(image.mode, (new_width, new_height), (255, 255, 255))
            scar = apply_jittering(scar, augs)
            scar_with_pad.paste(scar, (left, top))
        else:
            scar_with_pad = Image.new(image.mode, (scar_w, scar_h), (255, 255, 255))
            scar = apply_jittering(scar, augs)
            scar_with_pad.paste(scar, (0, 0))
        scar = scar_with_pad.convert('RGBA')
    angle = random.randint(-45, 45)
    scar = scar.rotate(angle, expand=True)

    #posizione casuale della sezione
    if (container.right-scar_w) > container.left:
        left = random.randint(container.left, container.right-scar_w)
    else:
        left = container.left
    if (container.bottom-scar_h) > container.top:
        top = random.randint(container.top, container.bottom-scar_h)
    else:
        top = container.top
    #left, top = random.randint(0, img_w - scar_w), random.randint(0, img_h - scar_h)
    return scar, (left, top)


def generate_rotations(image):
    r90 = image.rotate(90)
    r180 = image.rotate(180)
    r270 = image.rotate(270)
    return image, r90, r180, r270


def generate_rotation(image):
    rotation = random.choice([0, 90, 180, 270])
    return image.rotate(rotation)


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


def generate_patch(
        image, 
        area_ratio:tuple=(0.02, 0.15), 
        aspect_ratio:tuple=((0.3, 1),(1, 3.3)),
        polygoned=False,
        distortion=False):

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
        mask = Image.new('RGBA', (patch_w, patch_h), (255,255,255,0)) 
        draw = ImageDraw.Draw(mask)
        
        points = get_random_points(
            mask.size[0],
            mask.size[1],
            5,
            15)
        draw.polygon(points, fill='black')
        
    if distortion:
        deformer = Deformer(imsize=image.size, points=(patch_left, patch_top, patch_right, patch_bottom))
        deformed_image = ImageOps.deform(image, deformer)
        cropped_patch = deformed_image.crop((patch_left, patch_top, patch_right, patch_bottom))
    else:
        cropped_patch = image.crop((patch_left, patch_top, patch_right, patch_bottom))
    return cropped_patch, mask, (paste_left, paste_top)


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


def generate_scar_new(image, w_range=(2,16), h_range=(10,25), augs=None, with_padding=False):
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
    if with_padding:
        scar_with_pad = Image.new(image.mode, (new_width, new_height), (255, 255, 255))
        scar = apply_jittering(scar, augs)
        scar_with_pad.paste(scar, (left, top))
    else:
        scar_with_pad = Image.new(image.mode, (scar_w, scar_h), (255, 255, 255))
        scar = apply_jittering(scar, augs)
        scar_with_pad.paste(scar, (0, 0))
    scar = scar_with_pad.convert('RGBA')
    angle = random.randint(-45, 45)
    scar = scar.rotate(angle, expand=True)

    #posizione casuale della sezione
    left, top = random.randint(0, img_w - scar_w), random.randint(0, img_h - scar_h)
    return scar, (left, top)


def generate_polygoned_scar(image, 
                  w_range:tuple=(2,16), 
                  h_range:tuple=(10,25),
                  augs=None,
                  colorized=True):
    img_w, img_h = image.size

    scar_w = random.randint(w_range[0], w_range[1])
    scar_h = random.randint(h_range[0], h_range[1])
        
    patch_left, patch_top = random.randint(0, img_w - scar_w), random.randint(0, img_h - scar_h)
    patch_right, patch_bottom = patch_left + scar_w, patch_top + scar_h

    #scar = Image.new('RGBA', (scar_w, scar_h), color=color)
    
    #scar_with_pad = Image.new(image.mode, (new_width, new_height), (255, 255, 255))
    
    mask = Image.new('RGBA', (scar_w, scar_h), (0,0,0,0)) 
    draw = ImageDraw.Draw(mask)
    points = 0.1 + 0.8*np.random.rand(random.randint(3,5), 2)
    x = [points[i][0] for i in range(len(points))]
    y = [points[i][1] for i in range(len(points))]
    x1 = normalize_in_interval(x, 0, mask.size[0])
    y1 = normalize_in_interval(y, 0, mask.size[1])
    points = [(x1[i], y1[i]) for i in range(len(points))]
    draw.polygon(points, fill='white')
    
    if colorized:
        r = random_color()
        g = random_color()
        b = random_color()
        color = (r,g,b)
        scar = Image.new('RGBA', (scar_w, scar_h), color=color)
    else:
        scar = image.crop((patch_left, patch_top, patch_right, patch_bottom))
        scar = apply_jittering(scar, augs)
    scar = scar.convert('RGBA')
    angle = random.randint(-45, 45)
    scar = scar.rotate(angle, expand=True)
    mask = mask.rotate(angle, expand=True)
    #posizione casuale della sezione
    left, top = random.randint(0, img_w - scar_w), random.randint(0, img_h - scar_h)
    return scar, mask, (left, top)
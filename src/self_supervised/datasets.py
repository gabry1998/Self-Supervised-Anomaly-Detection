from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageOps
from sklearn.model_selection import train_test_split as tts
from skimage.segmentation import slic
from skimage import color
from torch.utils.data import DataLoader, Dataset
from scipy.signal import savgol_filter
from torchvision import transforms
from torchvision.transforms import Compose
from self_supervised.dataset_generator import \
    check_color_similarity, \
    check_valid_coordinates_by_container, \
    generate_patch, \
    get_random_coordinate, \
    obj_mask, paste_patch, \
    rect2poly
from self_supervised.functional import \
    duplicate_filenames, \
    get_all_subject_experiments, \
    get_ground_truth, \
    get_ground_truth_filename, \
    get_filenames, \
    get_test_data_filenames
from numpy import ndarray
from self_supervised import constants
import random
import torch
import numpy as np
import pytorch_lightning as pl
from skimage.transform import swirl



class CPP:
    jitter_offset = 0.1
        
    rectangle_area_ratio_patch = (0.2, 0.5) # patch-wise
    rectangle_area_ratio = (0.03, 0.07) # image-wise
    rectangle_aspect_ratio = ((0.3, 0.5),(1, 3.3))

    scar_area_ratio_patch = (0.02, 0.05) # patch-wise
    scar_area_ratio = (0.003, 0.007) # image-wise
    scar_aspect_ratio = ((0.3, 0.5),(2.5, 3.3))

    jitter_transforms = transforms.ColorJitter(
                            brightness = jitter_offset,
                            contrast = jitter_offset,
                            saturation = jitter_offset)


class MVTecDataset(Dataset):
    def __init__(
            self,
            dataset_dir:str,
            images_filenames:list,
            imsize:tuple=(256,256),
            transform:Compose=None,
            patch_level:bool=False) -> None:
        
        super().__init__()
        self.dataset_dir = dataset_dir
        self.images_filenames = images_filenames
        self.imsize = imsize
        self.transform = transform
        self.patch_level = patch_level
            
    
    def __getitem__(self, index):
        filename = self.images_filenames[index]
        original = Image.open(filename).resize(self.imsize).convert('RGB')
        
        gt_filename = get_ground_truth_filename(
            filename,
            self.dataset_dir+'ground_truth/')
        gt = get_ground_truth(gt_filename, self.imsize)
        x = original.copy()
        if self.transform:
            x = self.transform(original)
        gt = transforms.ToTensor()(gt)
        
        return x, gt, transforms.ToTensor()(original)
        
        
    def __len__(self):
        return self.images_filenames.shape[0]


class MVTecDatamodule(pl.LightningDataModule):
    def __init__(
            self,
            root_dir:str, # something as ../dataset/bottle/
            imsize:tuple=(256,256),
            batch_size:int=32,  
            seed:int=0,
            patch_level=None):
            
        super().__init__()
        self.root_dir = root_dir
        self.imsize = imsize
        self.batch_size = batch_size
        self.seed = seed
        self.patch_level = patch_level
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        self.train_images_filenames = get_filenames(self.root_dir+'/train/good/')
        self.test_images_filenames = get_test_data_filenames(self.root_dir+'/test/')
    
    
    def setup(self, stage=None) -> None:
        train_images_filenames, val_images_filenames = tts(
            self.train_images_filenames, 
            test_size=0.2, 
            random_state=self.seed)
        
        self.train_dataset = MVTecDataset(
            self.root_dir,
            train_images_filenames,
            imsize=self.imsize,
            transform=self.transform,
        )
        self.val_dataset = MVTecDataset(
            self.root_dir,
            val_images_filenames,
            imsize=self.imsize,
            transform=self.transform,
        )
        self.test_dataset = MVTecDataset(
            self.root_dir,
            self.test_images_filenames,
            imsize=self.imsize,
            transform=self.transform,
        )
     
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=8)
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=8)
    
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=8)
    
    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=8)
 

class PretextTaskDataset(Dataset):
    def __init__(
            self,
            subject:str,
            images_filenames:ndarray,
            imsize:tuple=(256,256),
            transform:Compose=None,
            patch_localization:bool=False,
            patch_size:tuple=64) -> None:

        super(PretextTaskDataset).__init__()
        self.subject = subject
        self.images_filenames = images_filenames
        
        self.imsize = imsize
        self.transform = transform
        self.patch_localization = patch_localization
        self.patch_size = patch_size
        
        self.patch_area_ratio = CPP.rectangle_area_ratio_patch if patch_localization else CPP.rectangle_area_ratio
        self.scar_area_ratio = CPP.scar_area_ratio_patch if patch_localization else CPP.scar_area_ratio
        
        # load N images, for each object class in dataset
        all_classes = np.array(get_all_subject_experiments('dataset/'))
        self.images_for_cut = [
            Image.open('dataset/'+sub+'/train/good/000.png').resize(self.imsize).convert('RGB') \
                for sub in all_classes
        ]
        
        # create a single mask for position-fixed object
        # textures just have a white image
        if self.subject in constants.TEXTURES():
            self.fixed_segmentation = Image.new(size=self.imsize, mode='RGB', color='white')
        else:
            temp = Image.open('dataset/'+self.subject+'/train/good/000.png').resize(self.imsize).convert('RGB')
            if self.subject == 'cable':
                image_array = np.array(temp)
                segments = slic(image_array, n_segments = 5, sigma =2, convert2lab=True)
                superpixels = color.label2rgb(segments, image_array, kind='avg')
                temp = Image.fromarray(superpixels).convert('RGB')
            self.fixed_segmentation = obj_mask(temp)
        
    
    def __getitem__(self, index:int):
        # load image
        original = Image.open(
            self.images_filenames[index])
        original = original.resize(self.imsize).convert('RGB')
        # apply label
        y = random.randint(0, 3)
        
        # copy original for second use
        x = original.copy()
        
        if not self.patch_localization and self.subject not in constants.NON_FIXED_OBJECTS():
            affine = transforms.RandomAffine(3, scale=(1.05,1.1))
            x = affine(x)
            
        # get image to crop for artificial defect
        if self.subject in constants.TEXTURES():
            image_for_cutting:Image.Image = random.choice(self.images_for_cut)
        else:
            image_for_cutting = original
        
        
        # create new masks only for non-fixed objects
        if self.subject in constants.NON_FIXED_OBJECTS():
            segmentation = obj_mask(original)
        else:
            segmentation = self.fixed_segmentation
            
        # container dims
        container_scaling_factor_patch = 1.75
        container_scaling_factor_scar = 2
        
        # crop image if patch-level mode
        if self.patch_localization:
            if self.subject == 'capsule':
                x = x.crop((0,50,255,200))
                segmentation = segmentation.crop((0,50,255,200))
            if self.subject == 'screw':
                x = x.crop((25,25,230,230))
                segmentation = segmentation.crop((25,25,230,230))
            left = random.randint(0,x.size[0]-self.patch_size)
            top = random.randint(0,x.size[1]-self.patch_size)
            x = x.crop((left,top, left+self.patch_size, top+self.patch_size))
            segmentation = segmentation.crop((left,top, left+self.patch_size, top+self.patch_size))
            image_for_cutting = transforms.RandomCrop(self.patch_size)(image_for_cutting)
            # container dim equal to patch
            container_scaling_factor_patch = 1
            container_scaling_factor_scar = 1
            # check working area sizes
            if torch.sum(transforms.ToTensor()(segmentation)) < int((self.patch_size*self.patch_size)/2):
               y = 0
        
        if y > 0:
            # get coordinate map
            segmentation = np.array(segmentation.convert('1'))
            coords_map = np.flip(np.column_stack(np.where(segmentation == 1)), axis=1)
           
            # big defect (polygon)
            if y == 1:
                # random position inside object mask to paste artificial defect
                coords = get_random_coordinate(coords_map)
                t = np.random.choice([0,1,2], p=[0.7, 0.15, 0.15])
                if t == 0:
                    patch = generate_patch(
                        image_for_cutting,
                        area_ratio=self.patch_area_ratio,
                        aspect_ratio=CPP.rectangle_aspect_ratio,
                    )
                    
                if t == 1:
                    patch = generate_patch(
                        image_for_cutting,
                        area_ratio=self.patch_area_ratio,
                        aspect_ratio=CPP.rectangle_aspect_ratio,
                        colorized=True,
                        color_type='average'
                    )
                if t == 2:
                    patch = generate_patch(
                        image_for_cutting,
                        area_ratio=self.patch_area_ratio,
                        aspect_ratio=CPP.rectangle_aspect_ratio,
                        colorized=True,
                        color_type='random'
                    )
                #patch = CPP.jitter_transforms(patch)
                if check_color_similarity(x, patch) > 0.99:
                    low = np.random.uniform(0.75, 0.9)
                    high = np.random.uniform(1.1, 1.15)
                    patch = ImageEnhance.Brightness(patch).enhance(random.choice([low, high]))
                    patch = ImageEnhance.Brightness(patch).enhance(random.choice([low, high]))
                coords = check_valid_coordinates_by_container(
                        x.size, 
                        patch.size, 
                        current_coords=coords,
                        container_scaling_factor=container_scaling_factor_patch
                    )
                mask = None
                mask = rect2poly(patch, regular=False, sides=8)
                x = paste_patch(x, patch, coords, mask) 
            # small defect (scar)
            if y == 2:
                t = np.random.choice([0,1,2], p=[0.7, 0.15, 0.15])
                if t == 0:
                    scar = generate_patch(
                        image_for_cutting,
                        area_ratio=self.scar_area_ratio,
                        aspect_ratio=CPP.scar_aspect_ratio
                    )
                if t == 1:
                    scar = generate_patch(
                        image_for_cutting,
                        area_ratio=self.scar_area_ratio,
                        aspect_ratio=CPP.scar_aspect_ratio,
                        colorized=True,
                        color_type='average'
                    )
                if t == 2:
                    scar = generate_patch(
                        image_for_cutting,
                        area_ratio=self.scar_area_ratio,
                        aspect_ratio=CPP.scar_aspect_ratio,
                        colorized=True,
                        color_type='random'
                    )
                #scar = CPP.jitter_transforms(scar)
                if check_color_similarity(x, scar) > 0.99:
                    low = np.random.uniform(0.75, 0.9)
                    high = np.random.uniform(1.1, 1.15)
                    scar = ImageEnhance.Brightness(scar).enhance(random.choice([low, high]))
                    scar = ImageEnhance.Brightness(scar).enhance(random.choice([low, high]))
                scar = scar.convert('RGBA')
                k = random.randint(2,5)
                angle = random.randint(-45,45)
                
                s = scar.rotate(angle, expand=True)
                
                offset = min(scar.size)
                for _ in range(k):
                    coords = get_random_coordinate(coords_map)
                    coords = check_valid_coordinates_by_container(
                        x.size, 
                        s.size, 
                        current_coords=coords,
                        container_scaling_factor=container_scaling_factor_scar
                    )
                    x = paste_patch(x, s, coords, s)
            # long line
            if y == 3:
                draw = ImageDraw.Draw(x)
                side = random.choice(['left','top'])
                n = 60 if not self.patch_localization else 30
                points = []
                c = 0
                for i in range(n):
                    offset = i/n
                    index = random.randint(c, int(len(coords_map)*offset))
                    p = coords_map[index]
                    points.append(tuple(p))
                    c = index
                rgb = random.choice(['black','white','silver'])
                
                if side == 'left':
                    points.sort(key=lambda tup: tup[0])
                points = savgol_filter(points, 10, 2, axis=0)
                if not self.patch_localization:
                    p_splits = np.array_split(points, 10)
                    k = random.randint(0,9)
                    points = p_splits[k]
                #else:
                #    p_splits = np.array_split(points, 3)
                #    k = random.randint(0,2)
                #    points = p_splits[k]
                points = [tuple(x) for x in points]
                if self.patch_localization:
                    draw.line(
                    points, fill=rgb, width=1)
                else:
                    draw.line(
                        points, fill=rgb, width=3)
                
        # apply jittering
        x = CPP.jitter_transforms(x)
        if self.transform:
            x = self.transform(x)
        return x, y, transforms.ToTensor()(original)
    
    
    def __len__(self):
        return self.images_filenames.shape[0]
    

class PretextTaskDatamodule(pl.LightningDataModule):
    def __init__(
            self, 
            subject:str,
            root_dir:str, #qualcosa come ../dataset/bottle/
            imsize:tuple=(256,256),
            batch_size:int=32,  
            train_val_split:float=0.2,
            seed:int=0,
            min_dataset_length:int=1000,
            duplication:bool=True,
            patch_localization:bool=False,
            patch_size:tuple=64):
        
        super().__init__()
        self.save_hyperparameters()
        self.root_dir_train = root_dir+'/train/good/'
        self.root_dir_test = root_dir+'/test/good/'
        self.subject = subject
        self.imsize = imsize
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        
        self.seed = seed
        self.min_dataset_length = min_dataset_length
        self.duplication = duplication
        self.patch_localization = patch_localization
        self.patch_size = patch_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        self.prepare_filenames()


    def prepare_filenames(self):
        images_filenames = get_filenames(self.root_dir_train)

        train_images_filenames, val_images_filenames = tts(
            images_filenames, 
            test_size=self.train_val_split, 
            random_state=self.seed)
        test_images_filenames = get_filenames(self.root_dir_test)
        
        if self.duplication:
            self.train_images_filenames = duplicate_filenames(
                train_images_filenames,
                self.min_dataset_length)
            self.val_images_filenames = duplicate_filenames(
                val_images_filenames,
                self.min_dataset_length)
            
            self.test_images_filenames = duplicate_filenames(
                    test_images_filenames,
                    self.min_dataset_length)
            
        else:
            self.train_images_filenames = train_images_filenames
            self.val_images_filenames = val_images_filenames
            self.test_images_filenames = test_images_filenames
        
        np.random.shuffle(np.array(self.train_images_filenames))
        np.random.shuffle(np.array(self.val_images_filenames))
        np.random.shuffle(np.array(self.test_images_filenames))
    
    
    def prepare_data(self) -> None:
        pass
    
    
    def setup(self, stage:str=None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = PretextTaskDataset(
                self.subject,
                self.val_images_filenames,
                imsize=self.imsize,
                transform=self.transform,
                patch_localization=self.patch_localization,
                patch_size=self.patch_size)
            
            self.val_dataset = PretextTaskDataset(
                self.subject,
                self.train_images_filenames,
                imsize=self.imsize,
                transform=self.transform,
                patch_localization=self.patch_localization,
                patch_size=self.patch_size)
            
        if stage == 'test' or stage == 'predict' or stage is None:
            self.test_dataset = PretextTaskDataset(
                self.subject,
                self.test_images_filenames,
                imsize=self.imsize,
                transform=self.transform,
                patch_localization=self.patch_localization,
                patch_size=self.patch_size)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
            num_workers=8)


    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            drop_last=True,
            persistent_workers=True,
            num_workers=8)
    
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=8)
        
    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=8)

import numpy as np
import pytorch_lightning as pl
from PIL import Image
from sklearn.model_selection import train_test_split as tts
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import Compose
from .support import constants as CONST
from .support.dataset_generator import *
from .support.cutpaste_parameters import CPP
from .support.functional import *
from numpy import array
from numpy.random import permutation

# dataset per le vere immagini mvtec
class MVTecDataset(Dataset):
    def __init__(
            self,
            dataset_dir:str,
            subject:str,
            images_filenames:array,
            imsize:tuple=CONST.DEFAULT_IMSIZE(),
            transform:Compose=None
            ) -> None:
        
        super().__init__()
        self.dataset_dir = dataset_dir
        self.subject = subject
        self.images_filenames = images_filenames
        self.imsize = imsize
        self.transform = transform
    
    def __getitem__(self, index):
        filename = self.images_filenames[index]
        test_image = Image.open(filename).resize(self.imsize).convert('RGB')
        
        gt_filename = get_mvtec_gt_filename_counterpart(
            filename,
            self.dataset_dir+'ground_truth/')
        gt = ground_truth(gt_filename, self.imsize)
        
        if self.transform:
            test_image = self.transform(test_image)
        gt = transforms.ToTensor()(gt)
        return test_image, gt
    
    def __len__(self):
        return self.images_filenames.shape[0]


class MVTecDatamodule(pl.LightningDataModule):
    def __init__(
            self,
            root_dir:str, # something as ../dataset/bottle/
            subject:str,
            imsize:tuple=CONST.DEFAULT_IMSIZE(),
            batch_size:int=CONST.DEFAULT_BATCH_SIZE(),  
            seed:int=CONST.DEFAULT_SEED(),
            localization=False):
            
        super().__init__()
        self.root_dir = root_dir
        self.subject = subject
        self.imsize = imsize
        self.batch_size = batch_size
        self.seed = seed
        self.localization = localization
        
        if localization:
            self.transform = transforms.Compose([
                transforms.ToTensor()])
        else:
            self.transform = CONST.DEFAULT_TRANSFORMS()
        
        self.train_images_filenames = get_image_filenames(self.root_dir+'/train/good/')
        self.test_images_filenames = get_mvtec_test_images(self.root_dir+'/test/')
    
    
    def setup(self, stage=None) -> None:
        self.train_dataset = MVTecDataset(
            self.root_dir,
            self.subject,
            self.train_images_filenames,
            transform=self.transform
        )
        self.test_dataset = MVTecDataset(
            self.root_dir,
            self.subject,
            self.test_images_filenames,
            transform=self.transform
        )
    
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4)
    
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4)
 

# avanti con questo tipo di dataset
class GenerativeDataset(Dataset):
    def __init__(
            self,
            images_filenames:array,
            imsize=CONST.DEFAULT_IMSIZE(),
            transform=None,
            distortion=False,
            polygons=False,
            patch_localization=False,
            patch_size:tuple=CONST.DEFAULT_PATCH_SIZE()) -> None:

        super().__init__()
        self.images_filenames = images_filenames
        self.area_ratio = CPP.cutpaste_augmentations['patch']['area_ratio']
        self.aspect_ratio = CPP.cutpaste_augmentations['patch']['aspect_ratio']
        self.scar_width = CPP.cutpaste_augmentations['scar']['width']
        self.scar_thiccness = CPP.cutpaste_augmentations['scar']['thiccness']
        
        self.imsize = imsize
        self.transform = transform
        self.distortion = distortion
        self.patch_localization = patch_localization
        self.patch_size = patch_size
        self.polygoned = polygons
        
        self.labels = self.generate_labels()

 
    def generate_labels(self):
        length = self.images_filenames.shape[0]
        return np.array(np.random.uniform(0,3, length), dtype=int)


    def __getitem__(self, index):
        x = self.images_filenames[index]
        y = self.labels[index]
        #y = random.randint(0, 2)
        
        x = self.generate_cutpaste_3way(x, y)
        
        if self.transform:
            x = self.transform(x)
        return x, y
    
    
    def __len__(self):
        return self.images_filenames.shape[0]
    
    
    def generate_cutpaste_3way(self, x, y):
        x = Image.open(x).resize(self.imsize).convert('RGB')
        if self.patch_localization:
            x = transforms.RandomCrop(self.patch_size)(x)
        
        if y == 0:
            return x
        if y == 1:
            #x = generate_rotation(x)
            patch, mask, coords = generate_patch(
                image=x, 
                area_ratio=self.area_ratio, 
                aspect_ratio=self.aspect_ratio, 
                polygoned=self.polygoned, 
                distortion=self.distortion)
            patch = apply_jittering(patch, CPP.jitter_transforms)
            x = paste_patch(x, patch, coords, mask)
            return x
        if y == 2:
            #x = generate_rotation(x)
            patch, coords = generate_scar_new(
                x, 
                self.scar_width, 
                self.scar_thiccness, 
                CPP.jitter_transforms,
                with_padding=True)
            x = paste_patch(x, patch, coords, patch)
            return x


class GenerativeDatamodule(pl.LightningDataModule):
    def __init__(
            self, 
            root_dir:str, #qualcosa come ../dataset/bottle/
            imsize:tuple=CONST.DEFAULT_IMSIZE(),
            batch_size:int=CONST.DEFAULT_BATCH_SIZE(),  
            train_val_split:float=CONST.DEFAULT_TRAIN_VAL_SPLIT(),
            seed:int=CONST.DEFAULT_SEED(),
            min_dataset_length:int=1000,
            duplication=False,
            polygoned=False,
            distortion=False,
            patch_localization=False,
            patch_size:tuple=CONST.DEFAULT_PATCH_SIZE()):
        
        super().__init__()
        self.save_hyperparameters()
        self.root_dir_train = root_dir+'/train/good/'
        self.root_dir_test = root_dir+'/test/good/'
        self.imsize = imsize
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        
        self.seed = seed
        self.min_dataset_length = min_dataset_length
        self.duplication = duplication
        self.polygoned = polygoned
        self.distortion = distortion
        self.patch_localization = patch_localization
        self.patch_size = patch_size

        self.transform = CONST.DEFAULT_TRANSFORMS()
        
        self.prepare_filenames()


    def prepare_filenames(self):
        images_filenames = get_image_filenames(self.root_dir_train)

        train_images_filenames, val_images_filenames = tts(
            images_filenames, 
            test_size=self.train_val_split, 
            random_state=self.seed)
        test_images_filenames = get_image_filenames(self.root_dir_test)
        
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
    
    
    def prepare_data(self) -> None:
        pass
    
    
    def setup(self, stage:str=None) -> None:
        if stage == 'fit' or stage is None:
            
            self.val_dataset = GenerativeDataset(
                self.val_images_filenames,
                imsize=self.imsize,
                transform=self.transform,
                polygons=self.polygoned,
                distortion=self.distortion,
                patch_localization=self.patch_localization,
                patch_size=self.patch_size)
            
        if stage == 'test' or stage is None:
            self.test_dataset = GenerativeDataset(
                self.test_images_filenames,
                imsize=self.imsize,
                transform=self.transform,
                polygons=self.polygoned,
                distortion=self.distortion,
                patch_localization=self.patch_localization,
                patch_size=self.patch_size)


    def train_dataloader(self):
        self.train_dataset = GenerativeDataset(
                self.train_images_filenames,
                imsize=self.imsize,
                transform=self.transform,
                polygons=self.polygoned,
                distortion=self.distortion,
                patch_localization=self.patch_localization,
                patch_size=self.patch_size)
        
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=True,
            num_workers=CONST.DEFAULT_NUM_WORKERS())


    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            drop_last=True,
            num_workers=CONST.DEFAULT_NUM_WORKERS())
    
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=CONST.DEFAULT_NUM_WORKERS())

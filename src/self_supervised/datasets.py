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



# dataset per le vere immagini mvtec
class MVTecDataset(Dataset):
    def __init__(
            self,
            dataset_dir:str,
            subject:str,
            images_filenames,
            imsize:tuple=CONST.DEFAULT_IMSIZE(),
            transform:Compose=None,
            ) -> None:
        
        super().__init__()
        self.dataset_dir = dataset_dir
        self.subject = subject
        self.images_filenames = images_filenames
        self.imsize = imsize
        self.transform = transform
    
    def __getitem__(self, index):
        filename = self.images_filenames[index]
        x = Image.open(filename).resize(self.imsize).convert('RGB')
        gt_filename = get_mvtec_gt_filename_counterpart(
            filename,
            self.dataset_dir+'ground_truth/')
        gt = ground_truth(gt_filename, self.imsize)
        x_hat = x.copy()
        if self.transform:
            x_hat = self.transform(x)
        gt = transforms.ToTensor()(gt)
        return x_hat, gt, transforms.ToTensor()(x)
    
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

        #self.transform = transforms.ToTensor()
        self.transform = CONST.DEFAULT_TRANSFORMS()
        
        self.train_images_filenames = get_image_filenames(self.root_dir+'/train/good/')
        self.test_images_filenames = get_mvtec_test_images(self.root_dir+'/test/')
    
    
    def setup(self, stage=None) -> None:
        train_images_filenames, val_images_filenames = tts(
            self.train_images_filenames, 
            test_size=0.2, 
            random_state=self.seed)
        
        self.train_dataset = MVTecDataset(
            self.root_dir,
            self.subject,
            train_images_filenames,
            transform=self.transform
        )
        self.val_dataset = MVTecDataset(
            self.root_dir,
            self.subject,
            val_images_filenames,
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
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=4)
    
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=4)
    
    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=4)
 

# avanti con questo tipo di dataset
class PeraDataset(Dataset):
    def __init__(
            self,
            subject:str,
            images_filenames,
            imsize:tuple=(256,256),
            transform=None,
            polygons:bool=False,
            colorized_scar:bool=False,
            patch_localization:bool=False,
            patch_size:tuple=64,
            mode:str='test' #test, train
            ) -> None:

        super(PeraDataset).__init__()
        self.subject = subject
        self.images_filenames = images_filenames
        self.area_ratio = CPP.cutpaste_augmentations['patch']['area_ratio']
        self.aspect_ratio = CPP.cutpaste_augmentations['patch']['aspect_ratio']
        self.scar_width = CPP.cutpaste_augmentations['scar']['width']
        self.scar_thiccness = CPP.cutpaste_augmentations['scar']['thiccness']
        
        self.imsize = imsize
        self.transform = transform
        self.patch_localization = patch_localization
        self.patch_size = patch_size
        self.polygoned = polygons
        self.colorized_scar = colorized_scar
        self.mode = mode
        
        all_classes = np.array(get_all_subject_experiments('dataset/'))
        idx = np.where(all_classes == self.subject)
        self.classes = np.delete(all_classes, idx)
        self.fixed_segmentation = obj_mask(Image.open('dataset/'+self.subject+'/train/good/000.png').resize(imsize).convert('RGB'))

    def __getitem__(self, index):
        # load image
        original = Image.open(
            self.images_filenames[index]).resize(
                self.imsize).convert('RGB')
        # apply label
        y = random.randint(0, 2)
        # copy original for second use
        x = original.copy()
        
        # create new masks only for non-fixed objects
        if self.subject in np.array(['hazelnut', 'screw', 'metal_nut']):
            segmentation = obj_mask(x)
        else:
            segmentation = self.fixed_segmentation
        
        
        
        container_scaling_factor_patch = 2
        container_scaling_factor_scar = 2.5
        
        # crop image if patch-level mode
        if self.patch_localization:
            left = random.randint(0,x.size[0]-self.patch_size)
            top = random.randint(0,x.size[1]-self.patch_size)
            x = x.crop((left,top, left+self.patch_size, top+self.patch_size))
            segmentation = segmentation.crop((left,top, left+self.patch_size, top+self.patch_size))
            container_scaling_factor_patch = 1
            container_scaling_factor_scar = 1
            # only background -> no transformations
            if torch.sum(transforms.ToTensor()(segmentation)) == 0:
                y = 0
        
        # apply jittering
        x = CPP.jitter_transforms(x)
        
        if y > 0:
            # random position inside object mask  to paste artificial defect
            coords = get_random_coordinate(segmentation)
            # big defect (polygon)
            if y == 1:
                if self.subject in np.array(['carpet','grid','leather','tile','wood']):
                    random_subject = random.choice(self.classes)
                    cutting = Image.open(
                        'dataset/'+random_subject+'/train/good/000.png'
                    ).resize(self.imsize).convert('RGB')
                    patch = generate_patch(
                            cutting,
                            area_ratio=[0.02, 0.05],
                            aspect_ratio=self.aspect_ratio,
                            augs=CPP.jitter_transforms)
                else:
                    patch = generate_patch(
                            original,
                            area_ratio=[0.02, 0.09],
                            aspect_ratio=self.aspect_ratio,
                            augs=CPP.jitter_transforms) 
                coords, _ = check_valid_coordinates_by_container(
                        x.size, 
                        patch.size, 
                        current_coords=coords,
                        container_scaling_factor=container_scaling_factor_patch
                    )
                mask = None
                mask = rect2poly(patch, regular=False, sides=8)
                x = paste_patch(x, patch, coords, mask) 
            # small defect (scar)
            else:
                if self.subject in np.array(['carpet','grid','leather','tile','wood']):
                    random_subject = random.choice(self.classes)
                    cutting = Image.open(
                        'dataset/'+random_subject+'/train/good/000.png'
                    ).resize(self.imsize).convert('RGB')
                    scar= generate_scar(
                        cutting,
                        self.scar_width,
                        self.scar_thiccness,
                        colorized=False,
                        color_type='average' # random, average, sample
                    )
                else:
                    scar= generate_scar(
                        original,
                        self.scar_width,
                        self.scar_thiccness,
                        colorized=False,
                        color_type='average' # random, average, sample
                    ) 
                coords, _ = check_valid_coordinates_by_container(
                        x.size, 
                        scar.size, 
                        current_coords=coords,
                        container_scaling_factor=container_scaling_factor_patch
                    )
                scar = scar.filter(ImageFilter.SHARPEN)
                angle = random.randint(-45,45)
                scar = scar.rotate(angle, expand=True)
                x = paste_patch(x, scar, coords, scar)
        if self.transform:
            x = self.transform(x)
        return x, y, transforms.ToTensor()(original) 
    
    
    def __len__(self):
        return self.images_filenames.shape[0]
    

class GenerativeDatamodule(pl.LightningDataModule):
    def __init__(
            self, 
            subject:str,
            root_dir:str, #qualcosa come ../dataset/bottle/
            imsize:tuple=CONST.DEFAULT_IMSIZE(),
            batch_size:int=CONST.DEFAULT_BATCH_SIZE(),  
            train_val_split:float=CONST.DEFAULT_TRAIN_VAL_SPLIT(),
            seed:int=CONST.DEFAULT_SEED(),
            min_dataset_length:int=1000,
            duplication=False,
            polygoned=False,
            colorized_scar=False,
            patch_localization=False,
            patch_size:tuple=CONST.DEFAULT_PATCH_SIZE()):
        
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
        self.polygoned = polygoned
        self.colorized_scar=colorized_scar
        self.patch_localization = patch_localization
        self.patch_size = patch_size

        self.transform = CONST.DEFAULT_TRANSFORMS()
        #self.transform = transforms.ToTensor()
        
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
        
        np.random.shuffle(np.array(self.train_images_filenames))
        np.random.shuffle(np.array(self.val_images_filenames))
        np.random.shuffle(np.array(self.test_images_filenames))
    
    
    def prepare_data(self) -> None:
        pass
    
    
    def setup(self, stage:str=None) -> None:
        if stage == 'fit' or stage is None:
            #self.train_dataset = GenerativeDataset(
            self.train_dataset = PeraDataset(
                self.subject,
                self.val_images_filenames,
                imsize=self.imsize,
                transform=self.transform,
                polygons=self.polygoned,
                colorized_scar=self.colorized_scar,
                patch_localization=self.patch_localization,
                patch_size=self.patch_size,
                mode='train')
            
            #self.val_dataset = GenerativeDataset(
            self.val_dataset = PeraDataset(
                self.subject,
                self.train_images_filenames,
                imsize=self.imsize,
                transform=self.transform,
                polygons=self.polygoned,
                colorized_scar=self.colorized_scar,
                patch_localization=self.patch_localization,
                patch_size=self.patch_size,
                mode='test')
            
        if stage == 'test' or stage is None:
            #self.test_dataset = GenerativeDataset(
            self.test_dataset = PeraDataset(
                self.subject,
                self.test_images_filenames,
                imsize=self.imsize,
                transform=self.transform,
                polygons=self.polygoned,
                colorized_scar=self.colorized_scar,
                patch_localization=self.patch_localization,
                patch_size=self.patch_size,
                mode='test')


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            drop_last=True,
            num_workers=8)


    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            drop_last=True,
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

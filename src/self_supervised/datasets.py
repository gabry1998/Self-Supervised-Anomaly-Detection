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
            transform:Compose=None,
            mode:str='test'
            ) -> None:
        
        super().__init__()
        self.dataset_dir = dataset_dir
        self.subject = subject
        self.images_filenames = images_filenames
        self.imsize = imsize
        self.transform = transform
        self.mode = mode
    
    def __getitem__(self, index):
        if self.mode == 'test':
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
        else:
            filename = self.images_filenames[index]
            train_image = Image.open(filename).resize(self.imsize).convert('RGB')
            gt = ground_truth(None, self.imsize)
            if self.transform:
                train_image = self.transform(train_image)
            gt = transforms.ToTensor()(gt)
            return train_image, gt
    
    def __len__(self):
        return self.images_filenames.shape[0]


class MVTecDatamodule(pl.LightningDataModule):
    def __init__(
            self,
            root_dir:str, # something as ../dataset/bottle/
            subject:str,
            imsize:tuple=CONST.DEFAULT_IMSIZE(),
            batch_size:int=CONST.DEFAULT_BATCH_SIZE(),  
            seed:int=CONST.DEFAULT_SEED()):
            
        super().__init__()
        self.root_dir = root_dir
        self.subject = subject
        self.imsize = imsize
        self.batch_size = batch_size
        self.seed = seed
        
        self.transform = CONST.DEFAULT_TRANSFORMS()
        #self.transform = transforms.Compose([
        #    transforms.ToTensor()])
        
        self.train_images_filenames = get_image_filenames(self.root_dir+'/train/good/')
        self.test_images_filenames = get_mvtec_test_images(self.root_dir+'/test/')
    
    
    def setup(self, stage=None) -> None:
        self.train_dataset = MVTecDataset(
            self.root_dir,
            self.subject,
            self.train_images_filenames,
            transform=self.transform,
            mode='train'
        )
        self.test_dataset = MVTecDataset(
            self.root_dir,
            self.subject,
            self.test_images_filenames,
            transform=self.transform,
            mode='test'
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
            transform=None) -> None:

        super().__init__()
        self.images_filenames = images_filenames
        self.area_ratio = CPP.cutpaste_augmentations['patch']['area_ratio']
        self.aspect_ratio = CPP.cutpaste_augmentations['patch']['aspect_ratio']
        self.scar_width = CPP.cutpaste_augmentations['scar']['width']
        self.scar_thiccness = CPP.cutpaste_augmentations['scar']['thiccness']
        
        self.imsize = imsize
        self.transform = transform
        
        self.labels = self.generate_labels()
   
 
    def generate_labels(self):
        length = self.images_filenames.shape[0]
        return np.array(np.random.uniform(0,3, length), dtype=int)


    def __getitem__(self, index):
        x = self.images_filenames[index]
        y = self.labels[index]
        x = self.generate_cutpaste_3way(x, y)
        
        if self.transform:
            x = self.transform(x)
        return x, y
    
    
    def __len__(self):
        return self.images_filenames.shape[0]
    
    
    def generate_cutpaste_3way(self, x, y):
        x = Image.open(x).resize(self.imsize).convert('RGB')
        x = generate_rotation(x)
        if y == 0:
            return x
        if y == 1:
            patch, coords = generate_patch(x, self.area_ratio, self.aspect_ratio)
            patch = apply_jittering(patch, CPP.jitter_transforms)
            x = paste_patch(x, patch, coords)
            return x
        if y == 2:
            patch, coords = generate_scar(x.size, self.scar_width, self.scar_thiccness)
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
            duplication=False):
        
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
                transform=self.transform)
            
        if stage == 'test' or stage is None:
            self.test_dataset = GenerativeDataset(
                self.test_images_filenames,
                imsize=self.imsize,
                transform=self.transform)


    def train_dataloader(self):
        self.train_dataset = GenerativeDataset(
                self.train_images_filenames,
                imsize=self.imsize,
                transform=self.transform)
        
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


# roba vecchia da rivedere
class CutPasteClassicDataset(Dataset):
    def __init__(
            self, 
            images, 
            labels, 
            transform=None, 
            target_transform=None):
        
        self.transform = transform
        self.target_transform = target_transform
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
  
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            image = torch.permute(image, (2, 1, 0))
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, int(label)
  
  
class CutPasteClassicDatamodule(pl.LightningDataModule):
    def __init__(
            self, 
            root_dir:str, #qualcosa come ../dataset/bottle/
            imsize=(256,256),
            batch_size:int=64,  
            train_val_split:float=0.2,
            classification_task='binary',
            seed:int=0,
            min_dataset_length=1000,
            duplication=False):
        
        super().__init__()
        self.save_hyperparameters()
        self.root_dir_train = root_dir+'/train/good/'
        self.root_dir_test = root_dir+'/test/good/'
        self.imsize = imsize
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.classification_task=classification_task
        
        self.seed = seed
        self.min_dataset_length = min_dataset_length
        self.duplication = duplication
        
        #random.seed(seed)
        #np.random.seed(seed)
        #torch.random.manual_seed(seed)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def prepare_data(self):
        pass      
    
    def setup(self, stage=None) -> None:
        if stage=='fit':
            training_data, training_labels = generate_dataset(
                self.root_dir_train,
                imsize=self.imsize,
                classification_task=self.classification_task
            )
            training_data, training_labels = list2np(
                training_data, 
                training_labels
            )
            
            training_data, training_labels = np2tensor(
                training_data, 
                training_labels
            )
            
            training_data, val_data, training_labels, val_labels = tts(
                training_data, 
                training_labels, 
                test_size=self.train_val_split,
                random_state=self.seed
            )
            
            self.train_dataset = CutPasteClassicDataset(
                training_data,
                training_labels,
                transform=self.transform
            )
            
            self.val_dataset = CutPasteClassicDataset(
                val_data,
                val_labels,
                transform=self.transform
            )
        
        if stage == 'test':
            test_data, test_labels = generate_dataset(
                self.root_dir_test,
                imsize=self.imsize,
                classification_task=self.classification_task
            )
            
            test_data, test_labels = list2np(
                test_data,
                test_labels
            )
            
            test_data, test_labels = np2tensor(
                test_data,
                test_labels
            )
            
            self.test_dataset = CutPasteClassicDataset(
                test_data,
                test_labels,
                transform=self.transform
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

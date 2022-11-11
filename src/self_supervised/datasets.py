import numpy as np
import pytorch_lightning as pl
from PIL import Image
from sklearn.model_selection import train_test_split as tts
from torch.utils.data import DataLoader, Dataset

from support.cutpaste_parameters import *
from support.dataset_generator import *
from support.filereader import get_image_filenames


class GenerativeDataset(Dataset):
    def __init__(
        self, 
        task,
        images_filenames,
        imsize=(256,256),
        transform=None) -> None:

        super().__init__()
        self.images_filenames = images_filenames

        self.augmentation_dict = augmentation_dict
        self.imsize = imsize
        self.task=task
        self.transform = transform
        
        self.labels = self.generate_labels()
    
    def generate_labels(self):
        length = self.images_filenames.shape[0]
        if self.task == '3-way':
            return np.array(np.random.uniform(0,3, length), dtype=int)
        if self.task == 'binary':
            return np.array(np.random.uniform(0,2, length), dtype=int)
        else:
            return np.empty(0)

    def __getitem__(self, index):
        x = self.images_filenames[index]
        y = self.labels[index]

        if self.task == '3-way':
            x = self.generate_cutpaste_3way(x, y)
        if self.task == 'binary':
            x = self.generate_cutpaste_binary(x, y)
        
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
            area_ratio = augmentation_dict['patch']['area_ratio']
            aspect_ratio = augmentation_dict['patch']['aspect_ratio']
            patch, coords = generate_patch(x, area_ratio, aspect_ratio)
            patch = apply_jittering(patch, augs)
            x = paste_patch(x, patch, coords)
            return x
        if y == 2:
            scar_width = augmentation_dict['scar']['width']
            scar_thiccness = augmentation_dict['scar']['thiccness']
            patch, coords = generate_scar(x.size, scar_width, scar_thiccness)
            x = paste_patch(x, patch, coords, patch)
            return x

    def generate_cutpaste_binary(self, x, y):
        x = Image.open(x).resize(self.imsize).convert('RGB')
        x = generate_rotation(x)
        if y == 0:
            return x
        else:
            if random.randint(0,1) == 1:
                area_ratio = augmentation_dict['patch']['area_ratio']
                aspect_ratio = augmentation_dict['patch']['aspect_ratio']
                patch, coords = generate_patch(x, area_ratio, aspect_ratio)
                patch = apply_jittering(patch, augs)
                x = paste_patch(x, patch, coords)
                return x
            else:
                scar_width = augmentation_dict['scar']['width']
                scar_thiccness = augmentation_dict['scar']['thiccness']
                patch, coords = generate_scar(x.size, scar_width, scar_thiccness)
                x = paste_patch(x, patch, coords, patch)
                return x

class GenerativeDatamodule(pl.LightningDataModule):
    def __init__(
            self, 
            root_dir:str, #qualcosa come ../dataset/bottle/
            imsize=(256,256),
            batch_size:int=64,  
            train_val_split:float=0.2,
            seed:int=0,
            n_repeat=1,
            classification_task='binary'):
        
        super().__init__()
        self.save_hyperparameters()
        self.root_dir_train = root_dir+'/train/good/'
        self.root_dir_test = root_dir+'/test/good/'
        self.imsize = imsize
        self.n_repeat = n_repeat
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.seed = seed
        self.classification_task = classification_task

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def prepare_data(self) -> None:
        pass
    
    def setup(self, stage:str=None) -> None:
        if stage == 'fit' or stage is None:
            images_filenames = get_image_filenames(
            self.root_dir_train, 
            n_repeat=self.n_repeat)

            self.train_images_filenames, self.val_images_filenames = tts(
                images_filenames, 
                test_size=self.train_val_split, 
                random_state=self.seed)
            
            self.val_dataset = GenerativeDataset(
                self.classification_task,
                self.val_images_filenames,
                imsize=self.imsize,
                transform=self.transform)
            
        if stage == 'test' or stage is None:
            self.test_images_filenames = get_image_filenames(
                self.root_dir_test, 
                n_repeat=self.n_repeat)
            
            self.test_dataset = GenerativeDataset(
                self.classification_task,
                self.test_images_filenames,
                imsize=self.imsize,
                transform=self.transform)

    def train_dataloader(self):
        self.train_dataset = GenerativeDataset(
                self.classification_task,
                self.train_images_filenames,
                imsize=self.imsize,
                transform=self.transform)
        
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
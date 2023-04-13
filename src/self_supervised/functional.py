from email.mime.text import MIMEText
from numpy import ndarray
from PIL import Image
from torch import Tensor
from typing import List
import numpy as np
import glob
import torch
import os
import smtplib



def get_all_subject_experiments(dataset_dir:str) -> List[str]:
    return sorted([name for name in os.listdir(dataset_dir) if os.path.isdir(
                        dataset_dir+name)
                    ])


def get_ground_truth(filename:str=None, imsize=(256,256)):
    if filename:
        return Image.open(filename).resize(imsize).convert('1')
    else:
        return Image.new(mode='1', size=imsize)


def get_prediction_class(predictions:Tensor) -> Tensor:
    y_hat = torch.max(predictions.data, 1)
    return y_hat.indices


def get_filenames(main_path:str) -> ndarray:
    filenames = sorted([f.replace("\\", '/') for f in glob.glob(main_path+'*.png', recursive = True)])
    return np.array(filenames)


def get_subdirectories(main_path:str) -> ndarray:
    return np.array(sorted([name for name in os.listdir(main_path) if os.path.isdir(
                        main_path+'/'+name)
                    ]), dtype=str)


def get_ground_truth_filename(test_filename:str, ground_truth_dir:str):
    filename_split = test_filename.rsplit('/', 2)
    defection = filename_split[1]
    image_name = filename_split[2]
    if defection == 'good':
        return None
    image_name = image_name.split('.')
    return ground_truth_dir+defection+'/'+image_name[0]+'_mask'+'.'+image_name[1]


def get_test_data_filenames(main_path:str) -> ndarray:
    anomaly_classes = get_subdirectories(main_path)
    test_images = np.empty(0)
    for defection in anomaly_classes:
        test_images = np.concatenate([
            test_images,
            get_filenames(main_path+defection+'/')
        ])
    return test_images


def duplicate_filenames(filenames:ndarray, baseline:int=2000) -> ndarray:
    dummy_copy = np.array(filenames, copy=True)
    while dummy_copy.shape[0] < baseline:
        dummy_copy = np.concatenate([dummy_copy, filenames], dtype=str)
    return dummy_copy


def extract_mask_patches(image:Tensor, dim:int=32, stride:int=4) -> Tensor:
    patches = image.unfold(2, dim, stride).unfold(3, dim, stride)
    patches = patches.reshape(-1,1, dim, dim)
    return patches


def extract_patches(image:Tensor, dim:int=32, stride:int=4) -> Tensor:
    b,c,h,w = image.shape
    patches = image.unfold(2, dim, stride).unfold(3, dim, stride)
    patches = patches.reshape(b, c, -1, dim, dim)
    patches = torch.permute(patches, (0,2,1,3,4))
    return patches


def normalize(tensor:Tensor) -> Tensor:
    tensor -= tensor.min()
    tensor /= tensor.max()
    return tensor


def normalize_in_interval(sample_mat:ndarray, interval_min:int, interval_max:int) -> ndarray:
    x =(sample_mat - np.min(sample_mat)) / (np.max(sample_mat) - np.min(sample_mat)) * (interval_max - interval_min) + interval_min
    x = np.rint(x)
    return x
    
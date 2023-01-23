import numpy as np
import glob
import torch
import os
from PIL import Image
from torch import Tensor
from scipy import signal
import torch.nn.functional as F



def get_all_subject_experiments(dataset_dir:str):
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


def get_image_filenames(main_path:str):
    return np.array(sorted([f.replace("\\", '/') for f in glob.glob(main_path+'*.png', recursive = True)]))


def get_subdirectories(main_path:str):
    return np.array([name for name in os.listdir(main_path) if os.path.isdir(
                        main_path+'/'+name)
                    ])


def get_ground_truth_filename(filename:str, groundtruth_dir):
    filename_split = filename.rsplit('/', 2)
    defection = filename_split[1]
    image_name = filename_split[2]
    if defection == 'good':
        return None
    image_name = image_name.split('.')
    return groundtruth_dir+defection+'/'+image_name[0]+'_mask'+'.'+image_name[1]


def get_mvtec_test_images(main_path:str):
    anomaly_classes = get_subdirectories(main_path)
    test_images = np.empty(0)
    for defection in anomaly_classes:
        test_images = np.concatenate([
            test_images,
            get_image_filenames(main_path+defection+'/')
        ])
    return test_images


def duplicate_filenames(filenames, baseline=2000):
    dummy_copy = np.array(filenames, copy=True)
    while dummy_copy.shape[0] < baseline:
        dummy_copy = np.concatenate([dummy_copy, filenames], dtype=str)
    return dummy_copy


def extract_mask_patches(image:Tensor, dim=32, stride=4):
    patches = image.unfold(2, dim, stride).unfold(3, dim, stride)
    patches = patches.reshape(-1,1, dim, dim)
    #patches = patches.squeeze()
    return patches


def extract_patches(image:Tensor, dim=32, stride=4):
    patches = image.unfold(2, dim, stride).unfold(3, dim, stride)
    patches = patches.reshape(1, 3, -1, dim, dim)
    patches = patches.squeeze()
    patches = torch.permute(patches, (1,0,2,3))
    return patches


def normalize(tensor:Tensor):
    tensor -= tensor.min()
    tensor /= tensor.max()
    return tensor


def normalize_in_interval(sample_mat, interval_min, interval_max):
    x =(sample_mat - np.min(sample_mat)) / (np.max(sample_mat) - np.min(sample_mat)) * (interval_max - interval_min) + interval_min
    x = np.rint(x)
    return x


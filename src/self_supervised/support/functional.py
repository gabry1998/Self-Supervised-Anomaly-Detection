import numpy as np
import glob
import torch
import os
from PIL import Image


def ground_truth(filename:str=None, imsize=(256,256)):
    if filename:
        return Image.open(filename).resize(imsize).convert('1')
    else:
        return Image.new(mode='1', size=imsize)


def get_image_filenames(main_path:str):
    return np.array(sorted([f for f in glob.glob(main_path+'*.png', recursive = True)]))


def get_mvtec_anomaly_classes(main_path:str):
    return np.array([name for name in os.listdir(main_path) if os.path.isdir(
                        os.path.join(main_path, name))
                    ])


def get_mvtec_gt_filename_counterpart(filename:str, groundtruth_dir):
    filename_split = filename.rsplit('/', 2)
    defection = filename_split[1]
    image_name = filename_split[2]
    if defection == 'good':
        return None
    
    image_name = image_name.split('.')
    return groundtruth_dir+defection+'/'+image_name[0]+'_mask'+'.'+image_name[1]


def get_mvtec_test_images(main_path:str):
    anomaly_classes = get_mvtec_anomaly_classes(main_path)
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


def list2np(images, labels):
    x = np.array([np.array(a, dtype=np.float32) for a in images])
    y = np.array(labels, dtype=int)
    return x,y


def np2tensor(images, labels):
    images = torch.as_tensor(images, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=int)
    return images,labels
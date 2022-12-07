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

def ground_truth(filename:str=None, imsize=(256,256)):
    if filename:
        return Image.open(filename).resize(imsize).convert('1')
    else:
        return Image.new(mode='1', size=imsize)


def get_prediction_class(predictions:Tensor) -> Tensor:
    y_hat = torch.max(predictions.data, 1)
    return y_hat.indices


def get_image_filenames(main_path:str):
    return np.array(sorted([f.replace("\\", '/') for f in glob.glob(main_path+'*.png', recursive = True)]))


def get_mvtec_anomaly_classes(main_path:str):
    return np.array([name for name in os.listdir(main_path) if os.path.isdir(
                        main_path+'/'+name)
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


def gt2label(gt_list:Tensor, negative=0, positive=1) -> list:
    return [negative if torch.sum(x) == 0 else positive for x in gt_list]


def multiclass2binary(predictions:Tensor) -> Tensor:
    return torch.tensor([1 if x > 0 else 0 for x in predictions])


def list2np(images, labels) -> tuple:
    x = np.array([np.array(a, dtype=np.float32) for a in images])
    y = np.array(labels, dtype=int)
    return x,y


def np2tensor(images, labels)-> tuple:
    images = torch.as_tensor(images, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=int)
    return images,labels


def imagetensor2array(image_tensor:Tensor):
    return np.array(torch.permute(image_tensor, (1,2,0)))


def heatmap2mask(heatmap, threshold=0.7):
    return heatmap > threshold


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


def black2white(im):
    data = np.array(im)   # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T # Temporarily unpack the bands for readability

    # Replace white with red... (leaves alpha values alone...)
    black_areas = (red < 50) & (blue < 50) & (green < 50)
    data[..., :-1][black_areas.T] = (255, 255, 255) # Transpose back needed

    return Image.fromarray(data)


def white2black(im):
    data = np.array(im)   # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T # Temporarily unpack the bands for readability

    # Replace white with red... (leaves alpha values alone...)
    white_areas = (red > 200) & (blue > 200) & (green > 200)
    data[..., :-1][white_areas.T] = (0,0,0) # Transpose back needed

    return Image.fromarray(data)


class GaussianSmooth:
    def __init__(self, kernel_size=32, stride=4, std=None, device=None):
        self.kernel_size = kernel_size
        self.stride = stride
        self.std = self.kernel_size_to_std() if not std else std
        if device:
            self.device = device 
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    def kernel_size_to_std(self):
        return np.log10(0.45*self.kernel_size + 1) + 0.25 if self.kernel_size < 32 else 10

    def gkern(self):
    
        if self.kernel_size % 2 == 0:
            # if kernel size is even, signal.gaussian returns center values sampled from gaussian at x=-1 and x=1
            # which is much less than 1.0 (depending on std). Instead, sample with kernel size k-1 and duplicate center
            # value, which is 1.0. Then divide whole signal by 2, because the duplicate results in a too high signal.
            gkern1d = signal.gaussian(self.kernel_size - 1, std=self.std).reshape(self.kernel_size - 1, 1)
            gkern1d = np.insert(gkern1d, (self.kernel_size - 1) // 2, gkern1d[(self.kernel_size - 1) // 2]) / 2
        else:
            gkern1d = signal.gaussian(self.kernel_size, std=self.std).reshape(self.kernel_size, 1)
        gkern2d = np.outer(gkern1d, gkern1d)
        return torch.tensor(gkern2d)
    
    def upsample(self, X):
        tconv = torch.nn.ConvTranspose2d(
            1,
            1, 
            kernel_size=self.kernel_size, 
            stride=self.stride)
        
        tconv.weight.data = self.gkern().unsqueeze(0).unsqueeze(0).float()
        tconv.to(self.device)
        X = X.float().to(self.device)
        out = tconv(X).detach().cpu()
        return out
    
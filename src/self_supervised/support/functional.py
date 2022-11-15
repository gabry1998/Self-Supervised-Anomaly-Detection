import numpy as np
import glob
import torch
import os
from PIL import Image
from torch import Tensor
from sklearn.neighbors import KernelDensity
from sklearn.covariance import LedoitWolf



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


def extract_patches(image:Tensor, dim=64, stride=32):
    patches = image.unfold(2, dim, stride).unfold(3, dim, stride)
    patches = patches.reshape(1, 3, -1, dim, dim)
    patches = patches.squeeze()
    patches = torch.permute(patches, (1,0,2,3))
    return patches


class GDE():
    def fit(self, embeddings):
        self.kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(embeddings)
        
    def predict(self, embeddings):
        scores = self.kde.score_samples(embeddings)
        scores = -scores
        return scores

class GaussianDensityTorch(object):
    """Gaussian Density estimation similar to the implementation used by Ripple et al.
    The code of Ripple et al. can be found here: https://github.com/ORippler/gaussian-ad-mvtec.
    """
    def fit(self, embeddings):
        self.mean = torch.mean(embeddings, axis=0)
        self.inv_cov = torch.Tensor(LedoitWolf().fit(embeddings.cpu()).precision_,device="cpu")

    def predict(self, embeddings):
        distances = self.mahalanobis_distance(embeddings, self.mean, self.inv_cov)
        return distances

    @staticmethod
    def mahalanobis_distance(
        values: torch.Tensor, mean: torch.Tensor, inv_covariance: torch.Tensor
    ) -> torch.Tensor:
        """Compute the batched mahalanobis distance.
        values is a batch of feature vectors.
        mean is either the mean of the distribution to compare, or a second
        batch of feature vectors.
        inv_covariance is the inverse covariance of the target distribution.
        from https://github.com/ORippler/gaussian-ad-mvtec/blob/4e85fb5224eee13e8643b684c8ef15ab7d5d016e/src/gaussian/model.py#L308
        """
        assert values.dim() == 2
        assert 1 <= mean.dim() <= 2
        assert len(inv_covariance.shape) == 2
        assert values.shape[1] == mean.shape[-1]
        assert mean.shape[-1] == inv_covariance.shape[0]
        assert inv_covariance.shape[0] == inv_covariance.shape[1]

        if mean.dim() == 1:  # Distribution mean.
            mean = mean.unsqueeze(0)
        x_mu = values - mean  # batch x features
        # Same as dist = x_mu.t() * inv_covariance * x_mu batch wise
        dist = torch.einsum("im,mn,in->i", x_mu, inv_covariance, x_mu)
        return dist.sqrt()
from torch import Tensor
import torch
import numpy as np



def gt2label(gt_list:Tensor, negative:int=0, positive:int=1) -> list:
    return [negative if torch.sum(x) == 0 else positive for x in gt_list]


def multiclass2binary(labels:Tensor) -> Tensor:
    return torch.tensor([1 if x > 0 else 0 for x in labels])


def imagetensor2array(image_tensor:Tensor, integer=True) -> np.ndarray:
    if integer:
        return np.array(torch.permute(image_tensor, (1,2,0))*255).astype(np.uint8)
    return np.array(torch.permute(image_tensor, (1,2,0)))

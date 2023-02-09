import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from self_supervised.models import PeraNet


x = torch.tensor([[1,2,3], [4,5,6]])
print(x ,x.shape)
a,b = x.shape
y = torch.reshape(x, ())
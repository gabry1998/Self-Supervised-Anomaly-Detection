import torch
from self_supervised import models
from torchvision import transforms
from torchvision.transforms import functional
from PIL import Image
from self_supervised.converters import imagetensor2array
from self_supervised.functional import extract_patches, get_prediction_class, normalize
from torch.nn import functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np

train_img = Image.open('dataset/capsule/train/good/000.png').resize((256,256)).convert('RGB')
test_img = Image.open('dataset/capsule/test/crack/001.png').resize((256,256)).convert('RGB')
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
train_tensor = transform(train_img)
test_tensor = transform(test_img)
train_patches = extract_patches(train_tensor[None,:], 32,4)
test_patches = extract_patches(test_tensor[None,:], 32,8)

model = models.PeraNet.load_from_checkpoint('brutta_copia/outputs/patch_level/computations/capsule/best_model.ckpt')
model.eval()
if torch.cuda.is_available():
    model.to('cuda')
if torch.cuda.is_available():
    with torch.no_grad():
        output = model(train_patches.to('cuda'))
        test_output = model(test_patches.to('cuda'))
else:
    with torch.no_grad():
        output = model(train_patches)
        test_output = model(test_patches)
detector = models.AnomalyDetector()

y_hats = get_prediction_class(output['classifier']).to('cpu')
patches_embeddings = output['latent_space'].to('cpu')
tot_embeddings = []
for i in range(len(patches_embeddings)):
    if y_hats[i] == 0:
        tot_embeddings.append(np.array(patches_embeddings[i]))
tot_embeddings = torch.tensor(np.array(tot_embeddings))

detector.fit(patches_embeddings)
tests = test_output['latent_space'].to('cpu')
anomaly_scores = detector.predict(tests)
anomaly_scores = normalize(anomaly_scores)
print(detector.threshold)

dim = int(np.sqrt(tests.shape[0]))       
saliency_map = torch.reshape(anomaly_scores, (dim, dim))
ksize = 3
saliency_map = functional.gaussian_blur(saliency_map[None,:], kernel_size=ksize).squeeze()
saliency_map = F.relu(saliency_map)
saliency_map = F.interpolate(saliency_map[None,None,:], 256, mode='bilinear').squeeze()
#saliency_map = cv2.applyColorMap(np.uint8(255 * saliency_map.squeeze()), cv2.COLORMAP_JET)
plt.imshow(np.array(saliency_map))
plt.savefig('a.png')
plt.close()
plt.imshow(test_img)
plt.savefig('b.png')
plt.close()
plt.imshow(np.array(saliency_map > detector.threshold))
plt.savefig('c.png')
plt.close()

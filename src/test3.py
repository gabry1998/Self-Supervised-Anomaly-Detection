from self_supervised.datasets import GenerativeDatamodule
from self_supervised.support.functional import *
import matplotlib.pyplot as plt
import numpy as np


def save_fig(my_array, name):
    hseparator = Image.new(mode='RGB', size=(6,256), color=(255,255,255))
    my_array = np.hstack([np.hstack(
      [np.array(my_array[i]), np.array(hseparator)]
      ) if i < len(my_array)-1 else np.array(my_array[i]) for i in range(len(my_array))])
    plt.figure(figsize=(30,30))
    plt.imshow(my_array)
    plt.axis('off')
    plt.savefig(name, bbox_inches='tight')
    plt.close()
    

dm = GenerativeDatamodule(
    'dataset/screw/',
    imsize=(256,256),
    batch_size=32,
    train_val_split=0.2,
    seed=0,
    polygoned=True,
    colorized_scar=True
)
dm.setup()
imgs = []
x, y = next(iter(dm.train_dataloader()))
for i in range(10):
    print(y[i])
    img = imagetensor2array(x[i], True)
    imgs.append(img)

save_fig(imgs, 'nonso.png')


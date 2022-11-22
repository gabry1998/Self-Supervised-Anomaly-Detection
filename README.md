# Self-Supervised Learning for Anomaly Detection

## Introduction
This project thesis aims to implement an Anomaly Detection framework using a Self-Supervised approach.


Self-supervised learning (SSL) is a subcategory of unsupervised learning. This method can achieve an excellent performance comparable to the fully-supervised baselines in several challenging tasks such as visual representation learning, object detection, and natural language processing.


SSL learns a generalizable representation from unlabelled data by solving a supervised proxy task (called PRETEXT TASK) which is often unrelated to the target task, but can help the network to learn a better representation.<br />
Pretext task examples:
* Rotation classification
* jigsaw puzzle
* denoising

The Pretext Task is solved creating an artificial dataset based on the original unlabelled dataset, where the model is trained to solve a problem in a supervised approach using pseudo labels (autogenerated during the dataset preparation, no human manual labeling).<br />
After training the network, the knowledge (weights) are transferred in a new model (equal or smaller) to solve the so-called Downstream Task (the original problem we wanted to solve), using the original real data.

## The Approach
This SSL approach is inspired by [CutPaste: Self-Supervised Learning for Anomaly Detection and Localization](http://openaccess.thecvf.com/content/CVPR2021/papers/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.pdf), where the pretext task is a classification task over an artificial dataset on which they are applied image transformations.<br />

The goal is to generate artificial anomalies and train the model to recognize those representations for the downstream task (Anomaly detection)

The dataset used is the MVTec dataset, widely used for anomaly detection benchmark.<br />
The personal contribution aims to create a Generative Dataset in the Pretext Task, where the dataset is re-generated every N training epochs. This choice is jusyfied by the fact the training images for each MVTec object class are very few (especially the toothbrush one). This choice should avoid the overfit problem (at least partially) and bring more generalization to the model

## Pretext Task
The pretext task is considered a supervised classification task in such a way the dataset is labeled in 3 classes:
* 0 -> anomaly-free image
* 1 -> cut-paste image (an image where a random portion of it is cut and pasted in a random location)
* 2 -> scar (a random colored line is drawn over the image in a random position)

First the labels are generated, then the respective image transformations are applied.

### Image Parameters
Following the paper values, we have:

|                   | Area Ratio    | Aspect Ratio       |
| :---------------: | :-----------: | :----------------: |
| CutPaste          | (0.02, 0.15)  | (0.3, 1) U (1, 3.3)|

|         | Width   | Height | Rotations |
| :-----: | :-----: | :----: | :-------: |
| Scar    | (2,16)  | (10,25)| (-45, 45) |

|  Jitter augmentations | Intensity |
| :--------------:      | :-----: |
|  Hue       | 0.1     |
|  Contrast       | 0.1     |
|  Brightness      | 0.1     |
|  Saturation       | 0.1     |

Those augmentations are only applied to the artificial anomalies (cutpaste patches) <br />
The images labeled as '0' are augmentation-free. <br />
All images have size (256,256) and are randomly rotated from a selection of [0, 90, 180, 270] degrees.

### CutPaste Examples

| Good |
| :--: |
| ![alt text](https://github.com/gabry1998/TesiAnomalyDetection/blob/main/readme_images/good.png) |

| CutPaste |
| :--: |
| ![alt text](https://github.com/gabry1998/TesiAnomalyDetection/blob/main/readme_images/cutpaste.png) |

| Scar |
| :--: |
| ![alt text](https://github.com/gabry1998/TesiAnomalyDetection/blob/main/readme_images/scar.png) |

### Network parameters

(some are not equal to the paper's one)
| |  |
| :-----: | :-----: |
| Batch size | 64 |
| seed | 0 |
| Backbone | ResNet-18 |
| Pretrained | Yes (Imagenet Weights) |
| Head projection | MPL of 5 layers with dim=512 (the 6-th has dim=128) |
| optimizer | SGD |
| learning rate | 0.001 |
| Momentum | 0.9 |
| Weight decay rate | 0.00003 |
| epochs (backbone frozen) | 30 |
| epochs (including backbone) | 20 |

### Computation Examples

#### Object (BOTTLES)

| bottle example |
| :--: |
| <img src="https://raw.githubusercontent.com/gabry1998/Self-Supervised-Anomaly-Detection/master/readme_images/bottle.png" width="256" height="256"/> |

| t-SNE | ROC |
| :--: | :--: |
| <img src="https://github.com/gabry1998/Self-Supervised-Anomaly-Detection/blob/main/outputs/computations/bottle/tsne.png"/> | <img src="https://github.com/gabry1998/Self-Supervised-Anomaly-Detection/blob/main/outputs/computations/bottle/roc.png"/>|

#### Texture (GRID)
| grid example |
| :--: |
|<img src="https://raw.githubusercontent.com/gabry1998/Self-Supervised-Anomaly-Detection/master/readme_images/grid.png" width="256" height="256"/> |

|t-SNE| ROC |
| :--: | :--: |
| <img src="https://github.com/gabry1998/Self-Supervised-Anomaly-Detection/blob/main/outputs/computations/grid/tsne.png"/> | <img src="https://github.com/gabry1998/Self-Supervised-Anomaly-Detection/blob/main/outputs/computations/grid/roc.png"/> |

#### Explanation
Textures defects are very hard to identify because of homogeneous patterns. Still, in the example we have an almost perfect scenario in BOTTLES, thanks to (assuming) easy object recognition in the images and his defects (real and artificial). In GRID we can see scars (2) isolated from the rest of classes, thanks to peculiarity of that defect (its literally a colored line over a homogeneous image). To improve Texture defect recognition we can apply more image augmentation, for example contrast, brightness, sharpening, etc to emphatize more the defect over the whole image and give the model a easier job.

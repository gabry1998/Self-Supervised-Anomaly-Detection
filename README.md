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
The personal contribution aims to create a variant of cutpaste approach, where instead of only cutting and pasting a rectangle over the image, we also create a binary mask of a random polygon and overlap the patch over the mask during the paste phase, to obtain an image with a polygonal shaped patch. The goal is to create a realistic approximation of real defects

## Pretext Task
The pretext task is considered a supervised classification task in such a way the dataset is labeled in 3 classes:
* 0 -> anomaly-free image
* 1 -> cut-paste image (an image where a random portion of it is cut and pasted in a random location)
* 2 -> scar (a random colored line is drawn over the image in a random position)

First the labels are generated, then the respective image transformations are applied.

### Image Parameters
(some are not equals to the paper's ones)

|                   | Area Ratio    | Aspect Ratio       |
| :---------------: | :-----------: | :----------------: |
| CutPaste          | (0.15, 0.30)  | (0.3, 1) U (1, 3.3)|

|         | Width   | Height | Rotations |
| :-----: | :-----: | :----: | :-------: |
| Scar    | (2,16)  | (10,25)| (-45, 45) |

|  Jitter augmentations | Intensity |
| :--------------:      | :-----: |
|  Hue       | 0.3     |
|  Contrast       | 0.3     |
|  Brightness      | 0.3     |
|  Saturation       | 0.3     |

Those augmentations are only applied to the artificial anomalies (cutpaste patches) <br />
The images labeled as '0' are augmentation-free. <br />
All images have size (256,256) and are randomly rotated from a selection of [0, 90, 180, 270] degrees.

### CutPaste (with polygon) Approach Example

<img src="https://raw.githubusercontent.com/gabry1998/Self-Supervised-Anomaly-Detection/master/outputs/dataset_analysis/screw/screw_artificial.png"/>

### Network parameters

(some are not equal to the paper's one)
| |  |
| :-----: | :-----: |
| Batch size | 96 |
| seed | 0 |
| Backbone | ResNet-18 |
| Pretrained | Yes (Imagenet Weights) |
| Head projection | MPL of 5 layers with dim=512 (the 6-th has dim=128) |
| optimizer | SGD |
| learning rate (backbone frozen) | 0.003 |
| learning rate (including backbone) | 0.001 |
| Momentum | 0.9 |
| Weight decay rate | 0.00003 |
| epochs (backbone frozen) | 30 |
| epochs (including backbone) | 20 |

The network is trained two times, first with imagenet weights frozen, then finetuning the whole Net.

### Computation Examples

#### Object (BOTTLES)

| t-SNE | ROC |
| :--: | :--: |
| <img src="https://raw.githubusercontent.com/gabry1998/Self-Supervised-Anomaly-Detection/master/outputs/computations/bottle/image_level/tsne.png"/> | <img src="https://raw.githubusercontent.com/gabry1998/Self-Supervised-Anomaly-Detection/master/outputs/computations/bottle/image_level/roc.png"/>|

Gradcam example (Image Level Localization)
<img src="https://raw.githubusercontent.com/gabry1998/Self-Supervised-Anomaly-Detection/master/outputs/computations/bottle/image_level/gradcam/heatmap_and_masks_0.png"/>

#### Texture (GRID)

|t-SNE| ROC |
| :--: | :--: |
| <img src="https://raw.githubusercontent.com/gabry1998/Self-Supervised-Anomaly-Detection/master/outputs/computations/grid/image_level/tsne.png"/> | <img src="https://raw.githubusercontent.com/gabry1998/Self-Supervised-Anomaly-Detection/master/outputs/computations/grid/image_level/roc.png"/> |

Gradcam example (Image Level Localization)
<img src="https://raw.githubusercontent.com/gabry1998/Self-Supervised-Anomaly-Detection/master/outputs/computations/grid/image_level/gradcam/heatmap_and_masks_0.png"/>

#### Explanation
In the example we have an almost perfect scenario in BOTTLES, and the real defect (purple) are in the same region as artificial defects. In GRID we can see a more sparse distribution, caused by the homogeneous patterns in images and small defects hard to localize, still, we can see real defect overlapped on scars and polygon patches
### Experiments results
Experiments run with 5 different seed and batch size of 64
|     | AUC | F1-SCORE | AUPRO |
| :--:     | :--:   | :--: | :--: |
average    |  0.87  |	0.77 |	0.46 
bottle     |	1	|   0.98 |  0.7 
cable      |  0.8	|   0.76 |	0.52 
capsule    |  0.74  |	0.44 |	0.26 
grid	   |  0.91  |	0.89 |	0.5 
metal nut  |  0.96  |	0.87 |	0.44 
screw      |  0.65  |	0.43 |	0.22 
tile       |  0.97  |	0.85 |	0.53 
toothbrush |  0.85  |	0.81 |	0.4 
zipper     |  0.97  |	0.91 |  0.56
### Other Examples

<img src="https://raw.githubusercontent.com/gabry1998/Self-Supervised-Anomaly-Detection/master/outputs/computations/cable/image_level/gradcam/heatmap_and_masks_1.png"/>
<img src="https://raw.githubusercontent.com/gabry1998/Self-Supervised-Anomaly-Detection/master/outputs/computations/capsule/image_level/gradcam/heatmap_and_masks_1.png"/>
<img src="https://raw.githubusercontent.com/gabry1998/Self-Supervised-Anomaly-Detection/master/outputs/computations/screw/image_level/gradcam/heatmap_and_masks_1.png"/>
<img src="https://raw.githubusercontent.com/gabry1998/Self-Supervised-Anomaly-Detection/master/outputs/computations/tile/image_level/gradcam/heatmap_and_masks_1.png"/>
<img src="https://raw.githubusercontent.com/gabry1998/Self-Supervised-Anomaly-Detection/master/outputs/computations/toothbrush/image_level/gradcam/heatmap_and_masks_0.png"/>
<img src="https://raw.githubusercontent.com/gabry1998/Self-Supervised-Anomaly-Detection/master/outputs/computations/zipper/image_level/gradcam/heatmap_and_masks_2.png"/> 

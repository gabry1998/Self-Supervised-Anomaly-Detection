import numpy as np
from torchvision import transforms


# options
def DATASET_GENERATION_TYPES():
    return np.array(['generative_dataset', 'classic_dataset'])


def CLASSIFICATION_TASK_TYPES():
    return np.array(['3-way', 'binary'])


def DEFAULT_CHECKPOINT_MODEL_NAME():
    return 'best_model.ckpt'


# dataset constants
def DEFAULT_CLASSIFICATION_TASK():
    return CLASSIFICATION_TASK_TYPES()[0]


def DEFAULT_DATASET_GENERATION():
    return DATASET_GENERATION_TYPES()[0]


def DEFAULT_IMSIZE():
    return (256,256)


def DEFAULT_PATCH_SIZE():
    return 64


def DEFAULT_NUM_WORKERS():
    return 8


# default model parameters
def DEFAULT_BATCH_SIZE():
    return 96


def DEFAULT_SEED():
    return 0


def DEFAULT_LEARNING_RATE():
    return 0.003


def DEFAULT_EPOCHS():
    return 30


def DEFAULT_TRAIN_VAL_SPLIT():
    return 0.2


def DEFAULT_PROJECTION_HEAD_DIMS():
    return np.array([512,512,512,512,512,512,512,512,512])


def DEFAULT_TRANSFORMS():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


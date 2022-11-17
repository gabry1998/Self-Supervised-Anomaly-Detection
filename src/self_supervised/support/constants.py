import numpy as np


def DATASET_GENERATION_TYPES():
    return np.array(['generative_dataset', 'classic_dataset'])


def CLASSIFICATION_TASK_TYPES():
    return np.array(['3-way', 'binary'])

# dataset constants
def DEFAULT_CLASSIFICATION_TASK():
    return CLASSIFICATION_TASK_TYPES()[0]


def DEFAULT_DATASET_GENERATION():
    return DATASET_GENERATION_TYPES()[0]


def DEFAULT_IMSIZE():
    return (256,256)

# default model parameters
def DEFAULT_BATCH_SIZE():
    return 64


def DEFAULT_SEED():
    return 0


def DEFAULT_LEARNING_RATE():
    return 0.001


def DEFAULT_EPOCHS():
    return 30


def DEFAULT_TRAIN_VAL_SPLIT():
    return 0.2


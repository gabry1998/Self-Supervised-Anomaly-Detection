from email.mime.text import MIMEText
from self_supervised.functional import get_all_subject_experiments
from training import run
from evaluator import evaluate
from datetime import datetime
import smtplib 
import numpy as np



def get_textures_names():
    return ['carpet','grid','leather','tile','wood']

def get_obj_names():
    return [
        'bottle',
        'cable',
        'capsule',
        'hazelnut',
        'metal_nut',
        'pill',
        'screw',
        'tile',
        'toothbrush',
        'transistor',
        'zipper'
    ]

def obj_set_one():
    return [
        'bottle',
        'cable',
        'capsule',
        'hazelnut',
        'metal_nut']

def obj_set_two():
    return [
        'pill',
        'screw',
        'toothbrush',
        'transistor',
        'zipper']

def specials():
    return [
        'cable',
        'capsule',
        'pill',
        'screw']


if __name__ == "__main__":
    patch_localization = True
    inputdir = 'brutta_copia/a/patch_level/computations/' if \
        patch_localization else 'brutta_copia/a/image_level/computations/'
    outputdir = 'brutta_copia/a/patch_level/computations/' if \
        patch_localization else 'brutta_copia/a/image_level/computations/'
    experiments = get_all_subject_experiments('dataset/')
    textures = get_textures_names()
    obj1 = obj_set_one()
    obj2 = obj_set_two()
    
    #### modificare qui ####
    experiments_list = ['cable','metal_nut','transistor']
    experiments_list2 = ['cable','metal_nut','transistor']
    #### -------------- ####
    
    
    # start training
    run(
        experiments_list=experiments_list,
        dataset_dir='dataset/', 
        root_outputs_dir=outputdir,
        imsize=(256,256),
        patch_localization=patch_localization,
        batch_size=64,
        projection_training_lr=0.003,
        projection_training_epochs=30,
        fine_tune_lr=0.003,
        fine_tune_epochs=50
    )        
        
    # start evaluation
    tot, textures_scores, obj_scores = evaluate(
        dataset_dir='dataset/',
        root_inputs_dir=inputdir,
        root_outputs_dir=outputdir,
        imsize=(256,256),
        patch_dim = 32,
        stride=8,
        seed=123456789,
        patch_localization=patch_localization,
        experiments_list=experiments_list2,
        artificial_batch_size=128
    )
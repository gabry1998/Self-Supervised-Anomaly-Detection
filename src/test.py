from self_supervised.datasets import GenerativeDatamodule
from self_supervised.support.dataset_generator import generate_dataset
from self_supervised.support.functional import *
from self_supervised.support.cutpaste_parameters import CPP

def test1():
    dataset_dir = '/home/ubuntu/TesiAnomalyDetection/dataset/'
    subject = 'toothbrush'

    x = get_image_filenames(dataset_dir+subject+'/train/good/')
    y = get_image_filenames(dataset_dir+subject+'/test/good/')

    print(x.shape)
    print(y.shape)

    x = duplicate_filenames(x)
    y = duplicate_filenames(y)

    print(x.shape)
    print(y.shape)

def test2():
    dataset_dir = '/home/ubuntu/TesiAnomalyDetection/dataset/'
    subject = 'toothbrush'
    
    y = get_mvtec_anomaly_classes(dataset_dir+subject+'/test/')
    print(y)
    
    x = get_mvtec_test_images(dataset_dir+subject+'/test/')
    print(x)
    y_hat1 = get_mvtec_gt_filename_counterpart(x[0], dataset_dir+subject+'/ground_truth/')
    y_hat2 = get_mvtec_gt_filename_counterpart(x[-1], dataset_dir+subject+'/ground_truth/')
    
    print(y_hat1)
    print(y_hat2)
    

def test3():
    print(CPP.summary)

def test4():
    dataset_dir = '/home/ubuntu/TesiAnomalyDetection/dataset/'
    subject = 'bottle'
    x, y = generate_dataset(
        dataset_dir+subject+'/train/good/',
        classification_task='3-way',
        duplication=True
    )
    x, y = list2np(x, y)
    x, y = np2tensor(x, y)
    print(x.shape, y.shape)
    
    print(x[0])

test4()
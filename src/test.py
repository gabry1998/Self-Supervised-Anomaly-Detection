from self_supervised.datasets import GenerativeDatamodule
from self_supervised.support.dataset_generator import generate_dataset
from self_supervised.support.functional import *
from self_supervised.support.cutpaste_parameters import CPP
from tqdm import tqdm
import time


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

def test5():
    x = np.array([[1,2],[3,45]])
    print(len(x))

def test6():
    
    for i1 in tqdm(range(5),leave=False):
        print('tante cose belle')
        imsize=(256,256)
        batch_size = 64
        train_val_split = 0.2
        seed = 0
        lr = 0.001
        epochs = 30
        
        print('image size:', imsize)
        print('batch size:', batch_size)
        print('split rate:', train_val_split)
        print('seed:', seed)
        print('optimizer:', 'SGD')
        print('learning rate:', lr)
        print('epochs:', epochs)
    
        for i2 in tqdm(range(300), leave=False):    
            time.sleep(0.01)
        
        os.system('clear')

test6()

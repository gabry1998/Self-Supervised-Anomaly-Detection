from ssl_datasets.generative_dataset import PretextTaskGenerativeDatamodule

dataset_dir = '/home/ubuntu/TesiAnomalyDetection/dataset/'
subject = 'toothbrush'

datamodule = PretextTaskGenerativeDatamodule(
    dataset_dir+subject+'/',
    n_repeat=1,
    classification_task='binary')

datamodule.prepare_data()
datamodule.setup()

print(datamodule.train_images_filenames.shape)
print(datamodule.val_images_filenames.shape)
print(datamodule.test_images_filenames.shape)
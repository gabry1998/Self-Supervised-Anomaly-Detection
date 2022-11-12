from self_supervised.model import SSLM, SSLModel, MetricTracker
from self_supervised.datasets import *
import pytorch_lightning as pl
from sklearn.metrics import classification_report


model_dir = 'outputs/computations/grid/generative_dataset/3-way/grid.ckpt'
dataset_dir = 'dataset/grid/'
sslm = SSLM(3)
sslm = SSLM.load_from_checkpoint(model_dir, model=sslm.model)

random.seed(0)
np.random.seed(0)
datamodule = GenerativeDatamodule(
            dataset_dir,
            classification_task='3-way',
            min_dataset_length=1000,
            duplication=True
)
datamodule.setup('test')

x,y = next(iter(datamodule.test_dataloader())) 
y_hat, _ = sslm(x)

print(y_hat.shape)
print(y.shape)

print(y)
y_hat = torch.max(y_hat.data, 1)
y_hat = y_hat.indices
print(y_hat)

print(
    classification_report( 
        y,
        y_hat,
        labels=[0,1,2]
    )
)
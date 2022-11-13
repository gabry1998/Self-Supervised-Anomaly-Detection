from torch import nn
import torch
from torchvision import models
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import importlib.util
import sys
from pytorch_lightning.callbacks.progress import ProgressBarBase
from tqdm import tqdm

class SSLModel(nn.Module):
  def __init__(self, 
               num_classes,
               seed=0):
    super().__init__()
    self.seed = seed
    self.num_classes = num_classes
    #random.seed(seed)
    #np.random.seed(seed)
    #torch.random.manual_seed(seed)

    self.feature_extractor = getattr(models, 'resnet18')(weights="IMAGENET1K_V1")
    last_layer= list(self.feature_extractor.named_modules())[-1][0].split('.')[0]
    setattr(self.feature_extractor, last_layer, nn.Identity())

    self.feature_extractor.eval()
    for param in self.feature_extractor.parameters():
        param.requires_grad = False

    self.feature_extractor.fc = nn.Identity()
    self.projection_head = nn.Sequential(
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),

        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),

        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),

        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),

        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),

        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True)

        )
    self.classifier = nn.Linear(128, self.num_classes)
      
  def compute_features(self, x):
      x = x.float()
      x = self.feature_extractor.conv1(x)
      x = self.feature_extractor.bn1(x)
      x = self.feature_extractor.relu(x)
      x = self.feature_extractor.maxpool(x)

      l1 = self.feature_extractor.layer1(x)
      l2 = self.feature_extractor.layer2(l1)
      l3 = self.feature_extractor.layer3(l2)
      l4 = self.feature_extractor.layer4(l3)

      x = self.feature_extractor.avgpool(l4)
      x = torch.flatten(x, 1)
      return (x, l1, l2, l3, l4)


  def forward(self, x):
    x = x.float()
    features = self.feature_extractor(x)
    #features = features.view(features.size(0), -1)
    embeddings = self.projection_head(features)
    output = self.classifier(embeddings)
    return (output, embeddings)


class SSLM(pl.LightningModule):
    def __init__(
            self,
            task:str='binary',
            lr=0.001,
            seed=0):
        super(SSLM, self).__init__()

        self.save_hyperparameters()
        self.lr = lr
        self.task = task
        self.num_classes = 3 if task == '3-way' else 2
        self.seed = seed

        self.model = SSLModel(self.num_classes)

        #random.seed(seed)
        #np.random.seed(seed)
        #torch.random.manual_seed(seed)

    
    def forward(self, x):
        output, embeddings = self.model(x)
        return  output, embeddings
    
    def training_step(self, batch, batch_idx):    
        x, y = batch
        y_hat, _ = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)

        metrics = {"train_accuracy": acc, "train_loss": loss}
        self.log_dict(metrics,
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)

        metrics = {"val_accuracy": acc, "val_loss": loss}
        self.log_dict(metrics,
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True)

        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        
        metrics = {"test_accuracy": acc, "test_loss": loss}
        self.log_dict(metrics,
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True)
        
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x,y = batch
        y_hat, _ = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        return loss, acc

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat, _ = self.model(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9, weight_decay=0.00003)
        return optimizer


class MetricTracker(pl.Callback):
  def __init__(self):
    
    self.log_metrics = {
        'train':{
            'accuracy':[],
            'loss':[]
        },
        'val':{
            'accuracy':[],
            'loss':[]
        }
    }

  def on_train_epoch_end(self, trainer, pl_module):
    elogs = trainer.logged_metrics
    self.log_metrics['train']['accuracy'].append(elogs['train_accuracy'].item())
    self.log_metrics['train']['loss'].append(elogs['train_loss'].item())
    self.log_metrics['val']['accuracy'].append(elogs['val_accuracy'].item())
    self.log_metrics['val']['loss'].append(elogs['val_loss'].item())

from torch import nn
import torch
from torchvision import models
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import random
import numpy as np
from sklearn.neighbors import KernelDensity
import self_supervised.support.constants as CONST
from numpy import array
from torch import Tensor


class SSLModel(nn.Module):
    def __init__(self, 
                num_classes:int,
                seed:int=CONST.DEFAULT_SEED(),
                dims:array = CONST.DEFAULT_PROJECTION_HEAD_DIMS()):
        
        super().__init__()
        self.seed = seed
        self.num_classes = num_classes
        self.localization = False

        self.feature_extractor = self.setup_feature_extractor()
        self.projection_head = self.setup_projection_head(dims)
        self.classifier = nn.Linear(128, self.num_classes)
        
        self.gradients = None
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    
    def setup_projection_head(self, dims):
        proj_layers = []
        for d in dims[:-1]:
            proj_layers.append(nn.Linear(d,d, bias=False)),
            proj_layers.append((nn.BatchNorm1d(d))),
            proj_layers.append(nn.ReLU(inplace=True))
        embeds = nn.Linear(dims[-2], dims[-1], bias=self.num_classes > 0)
        proj_layers.append(embeds)
        
        projection_head = nn.Sequential(
            *proj_layers
        )
        return projection_head
    
    
    def setup_feature_extractor(self):
        fe = getattr(models, 'resnet18')(weights="IMAGENET1K_V1")
        last_layer= list(fe.named_modules())[-1][0].split('.')[0]
        setattr(fe, last_layer, nn.Identity())

        fe.eval()
        for param in fe.parameters():
            param.requires_grad = False

        fe.fc = nn.Identity()
        return fe
    
    
    def get_activations_gradient(self):
        return self.gradients
    
    
    def set_for_localization(self, p=False):
        if p:
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
        self.localization = p
    
        
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
    
        avg_pool = self.feature_extractor.avgpool(l4)
        
        return (avg_pool, l1, l2, l3, l4)


    def forward(self, x):
        x = x.float()
        features = self.feature_extractor(x)
        
        hooker = features
        
        if self.localization:
            def __extract_grad(grad):
                self.gradients = grad
            hooker.register_hook(__extract_grad)
        
        
        features = torch.flatten(features, 1)
        embeddings = self.projection_head(features)
        output = self.classifier(embeddings)
        return (output, embeddings)


class SSLM(pl.LightningModule):
    def __init__(
            self,
            classification_task:str=CONST.DEFAULT_CLASSIFICATION_TASK(),
            lr:float=CONST.DEFAULT_LEARNING_RATE(),
            seed:int=CONST.DEFAULT_SEED()):
        
        super(SSLM, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.classification_task = classification_task
        self.num_classes = 3 if classification_task== '3-way' else 2
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        
        self.model = SSLModel(self.num_classes)

    
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


class GDE():
    def __init__(self) -> None:
        pass
        
    def fit(self, embeddings:Tensor):
            self.kde = KernelDensity().fit(embeddings)
        
        
    def predict(self, embeddings:Tensor):
        scores = self.kde.score_samples(embeddings)
        norm = np.linalg.norm(scores)
        return scores/norm
from math import sqrt
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import KernelDensity
from torch import nn
import torch
from torchvision import models
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import numpy as np
import self_supervised.support.constants as CONST
from numpy import array
from torch import Tensor
from numpy import dot
from numpy.linalg import norm

from self_supervised.support.functional import get_prediction_class, gt2label


class SSLModel(nn.Module):
    def __init__(self, 
                num_classes:int,
                dims:array = CONST.DEFAULT_PROJECTION_HEAD_DIMS()):
        
        super().__init__()
        self.num_classes = num_classes
        
        self.feature_extractor = self.setup_feature_extractor()
        self.projection_head = self.setup_projection_head(dims)
        self.classifier = nn.Linear(dims[-1], self.num_classes)
        
        self.localization = False
        
    
    def setup_projection_head(self, dims):
        proj_layers = []
        for d in dims[:-1]:
            layer = nn.Linear(d,d, bias=False)
            proj_layers.append(layer),
            #proj_layers.append(nn.Dropout(0.10)),
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
        for param in fe.fc.parameters():
            param.requires_grad = True
        return fe
    
    
    def unfreeze_layers(self, p=True):
        for param in self.feature_extractor.parameters():
            param.requires_grad = p
        for param in self.projection_head.parameters():
            param.requires_grad = p
    
    
    def set_for_localization(self, p=True):
        for param in self.feature_extractor.parameters():
            param.requires_grad = p
        for param in self.projection_head.parameters():
            param.requires_grad = p
        self.localization = p
    
        
    def compute_features(self, x, layer='layer4'):
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
        if layer=='layer1':
            return l1
        if layer=='layer2':
            return l2
        if layer=='layer3':
            return l3
        if layer=='layer4':
            return l4
        return (avg_pool, l1, l2, l3, l4)


    def forward(self, x):
        x = x.float()
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        #embeddings = self.dropout(self.projection_head(features))
        embeddings = self.projection_head(features)
        output = self.classifier(embeddings)
        if self.localization:
            return output
        return (output, embeddings)


class SSLM(pl.LightningModule):
    def __init__(
            self,
            num_epochs:int=None,
            lr:float=CONST.DEFAULT_LEARNING_RATE(),
            dims=CONST.DEFAULT_PROJECTION_HEAD_DIMS()):
        
        super(SSLM, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_classes = 3
        self.num_epochs = num_epochs
        
        self.model = SSLModel(self.num_classes, dims)
        self.localization = False
        self.mvtec = False
            
    def unfreeze_layers(self, p=True):
        self.model.unfreeze_layers(p)
    
    
    def set_for_localization(self, p=True):
        self.model.set_for_localization(p)
        self.localization = p
    
    
    def forward(self, x):
        if self.localization:
            output = self.model(x)
            return  output
        
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
        outputs = {}
        x, y = batch
        y_hat,  embeds = self.model(x)
        y_hat = get_prediction_class(y_hat)
        outputs['y_hat'] = y_hat
        outputs['embedding'] = embeds
        if self.mvtec:
            outputs['y_true'] = torch.tensor(gt2label(y, 0, 1))
            outputs['y_true_tsne'] = torch.tensor(gt2label(y, 0, 3))
            outputs['groundtruth'] = y
        else:
            outputs['y_true'] = y
        return outputs


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9, weight_decay=0.00003)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.num_epochs)
        return [optimizer], [scheduler]


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


class CosineSimilarityEstimator():
    def __init__(self, embeddings) -> None:
        #shape of (batch, num_patches, embedding_size)
        self.data = embeddings
        
    def fit(self):
        self.average_good = torch.mean(self.data, axis=0)
        
    
    def _compute(self, a, b):
        return dot(a, b)/(norm(a)*norm(b))
    
    def predict(self, embeddings:Tensor):
        #shape of (num_patches, embedding_size)
        return torch.tensor([self._compute(embeddings[i], self.average_good[i]) for i in range(len(embeddings))])


class GDE():
    def __init__(self) -> None:
        pass
        
    def fit(self, embeddings:Tensor):
            self.kde = KernelDensity().fit(embeddings)
        
        
    def predict(self, embeddings:Tensor):
        scores = self.kde.score_samples(embeddings)
        norm = np.linalg.norm(-scores)
        return torch.tensor(-scores/norm)

      
class MahalanobisDistance(object):
    def fit(self, embeddings:Tensor, mean:Tensor):
        #self.mean = torch.mean(embeddings, axis=0)
        self.mean = mean
        self.inv_cov = torch.Tensor(LedoitWolf().fit(embeddings.cpu()).precision_,device="cpu")

    def predict(self, embeddings:Tensor):
        distances = self.mahalanobis_distance(embeddings, self.mean, self.inv_cov)
        return distances

    @staticmethod
    def mahalanobis_distance(
        values: torch.Tensor, mean: torch.Tensor, inv_covariance: torch.Tensor
    ) -> torch.Tensor:
        assert values.dim() == 2
        assert 1 <= mean.dim() <= 2
        assert len(inv_covariance.shape) == 2
        assert values.shape[1] == mean.shape[-1]
        assert mean.shape[-1] == inv_covariance.shape[0]
        assert inv_covariance.shape[0] == inv_covariance.shape[1]

        if mean.dim() == 1:  # Distribution mean.
            mean = mean.unsqueeze(0)
        x_mu = values - mean  # batch x features
        # Same as dist = x_mu.t() * inv_covariance * x_mu batch wise
        dist = torch.einsum("im,mn,in->i", x_mu, inv_covariance, x_mu)
        return dist.sqrt()
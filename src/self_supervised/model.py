from sklearn.covariance import LedoitWolf
from sklearn.neighbors import KernelDensity
from torch import nn
from torchsummary import summary as torch_summary
from numpy import linalg as LA
import torch
from torchvision import models
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import numpy as np
from torch import Tensor
from self_supervised.support.functional import get_prediction_class, gt2label


class PeraNet(pl.LightningModule):
    def __init__(
            self, 
            latent_space_dims:list=[512,512,512,512,512,512,512,512,512,512,512],
            backbone:str='resnet18',
            lr:float=0.01,
            num_epochs:int=20,
            num_classes:int=3) -> None:
        super(PeraNet, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_epochs = num_epochs
        self.backbone = backbone
        self.latent_space_dims = latent_space_dims
        
        
        self.feature_extractor = self.setup_backbone(backbone)
        self.latent_space = self.setup_latent_space(latent_space_dims)
        self.classifier = self.setup_classifier(latent_space_dims[-1], num_classes)

        self.mvtec = False
        self.num_classes = num_classes
    
    def summary(self):
        torch_summary(self.to('cpu'), (3,256,256), device='cpu')
        
    def setup_backbone(self, backbone:str) -> nn.Sequential:
        if backbone == 'resnet18':
            model = models.resnet18(weights="IMAGENET1K_V1")
        layers = list(model.children())[:-1]
        feature_extractor = nn.Sequential(*layers)
        return feature_extractor
    
    
    def setup_latent_space(self, dims:list) -> nn.Sequential:
        proj_layers = []
        for i in range(len(dims[:-1])-1):
            layer = nn.Linear(dims[i], dims[i+1], bias=False)
            proj_layers.append(layer),
            proj_layers.append(nn.Dropout(0.25))
            proj_layers.append((nn.BatchNorm1d(dims[i+1]))),
            proj_layers.append(nn.ReLU(inplace=True))
        embeds = nn.Linear(dims[-2], dims[-1], bias=True)
        proj_layers.append(embeds)
        projection_head = nn.Sequential(
            *proj_layers
        )
        return projection_head
    
    
    def setup_classifier(self, input_dim:int, num_classes:int) -> nn.Linear:
        return nn.Linear(input_dim, num_classes)
        
    
    def enable_mvtec_inference(self):
        self.mvtec = True
    
    def disable_mvtec_inference(self):
        self.mvtec = False

    def unfreeze_net(self, modules:list=['backbone', 'latent_space']):
        if 'backbone' in modules:
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
        if 'latent_space' in modules:
            for param in self.latent_space.parameters():
                param.requires_grad = True
    
    
    def freeze_net(self, modules:list=['backbone', 'latent_space']):
        if 'backbone' in modules:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.feature_extractor.eval()
        if 'latent_space' in modules:
            for param in self.latent_space.parameters():
                param.requires_grad = False
            self.latent_space.eval()


    def layer_activations(self, x, layer:str='layer4'):
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)
        x = self.feature_extractor.maxpool(x)

        l1 = self.feature_extractor.layer1(x)
        l2 = self.feature_extractor.layer2(l1)
        l3 = self.feature_extractor.layer3(l2)
        l4 = self.feature_extractor.layer4(l3)
        
        if layer=='layer1':
            return l1
        if layer=='layer2':
            return l2
        if layer=='layer3':
            return l3
        if layer=='layer4':
            return l4
        
        
    def forward(self, x):
        output = {}
        
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        embeddings = self.latent_space(features)
        y_hat = self.classifier(embeddings)
        
        output['classifier'] = y_hat
        output['latent_space'] = embeddings
        return output
    
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        
        outputs = self(x)
        y_hat = outputs['classifier']
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)

        metrics = {"train_accuracy": acc, "train_loss": loss}
        self.log_dict(metrics,
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True)
        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        
        outputs = self(x)
        y_hat = outputs['classifier']
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)

        metrics = {"val_accuracy": acc, "val_loss": loss}
        self.log_dict(metrics,
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True)

        return metrics


    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        if self.mvtec:
            y = torch.tensor(gt2label(y))
            if torch.cuda.is_available():
                y = y.to('cuda')
            else:
                y = y.to('cpu')
        
        outputs = self(x)
        y_hat = outputs['classifier']
        
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        
        metrics = {"test_accuracy": acc, "test_loss": loss}
        self.log_dict(metrics,
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True)
        
        return metrics
    
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = {
            'x':None,
            'x_prime':None,
            'y_true':None,
            'y_tsne':None,
            'y_hat':None,
            'groundtruth':None,
            'embedding':None
        }
        x_prime, groundtruths, x = batch
        if self.mvtec:
            y = torch.tensor(gt2label(groundtruths))
            y_tsne = torch.tensor(gt2label(groundtruths, negative=0, positive=self.num_classes))
            if torch.cuda.is_available():
                y = y.to('cuda')
                y_tsne = y_tsne.to('cuda')
            else:
                y_tsne = y_tsne.to('cpu')
                y_tsne = y_tsne.to('cpu')
            outputs['y_true'] = y
            outputs['y_tsne'] = y_tsne 
            outputs['groundtruth'] = groundtruths
        else:
            outputs['y_true'] = groundtruths
            outputs['y_tsne'] = groundtruths 
            outputs['groundtruth'] = None  
            
        predictions = self(x_prime)
        y_hat = get_prediction_class(predictions['classifier'])
        
        outputs['x'] = x
        outputs['x_prime'] = x_prime
        outputs['embedding'] = predictions['latent_space']
        outputs['y_hat'] = y_hat
        
        return outputs
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), self.lr, momentum=0.9, weight_decay=0.0005)
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
    def fit(self, embeddings:Tensor):
        self.mean = torch.mean(embeddings, axis=0)
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
        #m = torch.dot(x_mu, torch.matmul(torch.inverse(inv_covariance), x_mu))
        return dist.sqrt()


def cal_mahal_dist(matrix):
    matrix_center = np.mean(np.array(matrix), axis=0)
    delta = matrix - matrix_center
    
    # calculate the covariance matrix and its inverse matrix
    cov_matrix = np.cov(matrix, rowvar=False, ddof=1)
    cov_matrix_inv = LA.inv(cov_matrix)  
    
    # calculate the Mahalanobis distance between a single vector and the center of the dataset
    def md_vector(vector):        
        inner_prod = np.dot(vector, cov_matrix_inv)
        dist = np.sqrt(np.dot(inner_prod, vector))
        return dist

    mahal_dist = np.apply_along_axis(arr=delta, axis=1, func1d=md_vector)
    assert len(mahal_dist) == len(matrix)
    return torch.tensor(mahal_dist)

    
    
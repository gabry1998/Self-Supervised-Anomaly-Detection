from sklearn.neighbors import KernelDensity, NearestNeighbors
from torch import nn
from torchsummary import summary as torch_summary
from torchvision.models import ResNet
from torchvision import models
from torchmetrics.functional import accuracy
from torch import Tensor
from self_supervised.converters import gt2label
from self_supervised.functional import get_prediction_class
from typing import Tuple
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn.functional as F




class PeraNet(pl.LightningModule):
    def __init__(
            self,
            intermediate_outputs:list=['layer2','layer3'],
            latent_space_layers:int=3,
            latent_space_layers_base_dim:int=512,
            num_classes:int=3,
            memory_bank_dim:int=100) -> None:
        super(PeraNet, self).__init__()

        self.backbone = 'resnet18'    
        self.__build_net(
            intermediate_outputs=intermediate_outputs,
            latent_space_layers=latent_space_layers,
            latent_space_layers_base_dim=latent_space_layers_base_dim,
            num_classes=num_classes)

        self.mvtec = False
        self.num_classes = num_classes
        
        self.batches_outputs = []
        self.memory_bank_dim = memory_bank_dim
        self.memory_bank = torch.tensor([])
    
    
    def compile(self, learning_rate:float=0.03, epochs:int=30) -> None:
        self.lr = learning_rate
        self.num_epochs = epochs
        self.save_hyperparameters()
        
    
    def __build_net(
            self, 
            intermediate_outputs:list=None,
            latent_space_layers:int=3,
            latent_space_layers_base_dim:int=512,
            num_classes:int=3) -> None:
        
        # function for hooks
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook

        # setup backbone
        self.feature_extractor = self.__setup_backbone()
        
        # func for FC layers dim
        def get_dim(layers:list, base_dim:int):
            dim = base_dim
            if 'layer1' in layers:
                dim += 64
                self.feature_extractor.layer1.register_forward_hook(get_activation('layer1'))
            if 'layer2' in layers:
                dim += 128
                self.feature_extractor.layer2.register_forward_hook(get_activation('layer2'))
            if 'layer3' in layers:
                dim += 256
                self.feature_extractor.layer3.register_forward_hook(get_activation('layer3'))
            return dim

        dim = latent_space_layers_base_dim if intermediate_outputs is None \
            else get_dim(intermediate_outputs, latent_space_layers_base_dim)
        # setup latent space
        self.latent_space = self.__setup_latent_space(
            latent_space_layers=latent_space_layers, 
            dim=dim)
        # setup classifier
        self.classifier = self.__setup_classifier(
            input_dim=dim, 
            num_classes=num_classes)
     
     
    def __setup_backbone(self) -> ResNet:
        feature_extractor = models.resnet18(weights="IMAGENET1K_V1")
        last_layer = list(feature_extractor.named_modules())[-1][0].split('.')[0]
        setattr(feature_extractor, last_layer, nn.Identity())
        return feature_extractor
    
    
    def __setup_latent_space(self,latent_space_layers:int, dim:int) -> nn.Sequential:
        dims = [dim for _ in range(latent_space_layers)]
        proj_layers = []
        if len(dims) > 1:
            for i in range(0,len(dims)-1):
                inp = dims[i]
                out = dims[i]
                layer = nn.Linear(inp, out, bias=False)
                proj_layers.append(layer)
                proj_layers.append(nn.BatchNorm1d(dims[i]))
                proj_layers.append(nn.ReLU(inplace=True))
            proj_layers.append(nn.Linear(dims[-1], dims[-1], bias=True))
        else:
            proj_layers.append(nn.Linear(dims[-1], dims[-1], bias=True))
        projection_head = nn.Sequential(
            *proj_layers
        )
        return projection_head
    
    
    def __setup_classifier(self, input_dim:int, num_classes:int) -> nn.Linear:
        return nn.Linear(input_dim, num_classes)
       

    def summary(self) -> None:
        torch_summary(self.to('cpu'), (3,256,256), device='cpu')
     
    
    def enable_mvtec_inference(self) -> None:
        self.mvtec = True
    
    
    def disable_mvtec_inference(self) -> None:
        self.mvtec = False


    def clear_memory_bank(self) -> None:
        self.memory_bank = torch.tensor([])


    def unfreeze_net(self, modules:list=['backbone', 'latent_space']) -> None:
        if 'backbone' in modules:
            print('uh')
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
        if 'latent_space' in modules:
            for param in self.latent_space.parameters():
                param.requires_grad = True
    
    
    def freeze_net(self, modules:list=['backbone', 'latent_space']) -> None:
        if 'backbone' in modules:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.feature_extractor.eval()
        if 'latent_space' in modules:
            for param in self.latent_space.parameters():
                param.requires_grad = False
            self.latent_space.eval()


    def layer_activations(self, x, layers:list=['layer4']) -> list:
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)
        x = self.feature_extractor.maxpool(x)

        l1 = self.feature_extractor.layer1(x)
        l2 = self.feature_extractor.layer2(l1)
        l3 = self.feature_extractor.layer3(l2)
        l4 = self.feature_extractor.layer4(l3)
        
        activations = []
        
        if 'layer1' in layers:
            activations.append(l1)
        if 'layer2' in layers:
            activations.append(l2)
        if 'layer3' in layers:
            activations.append(l3)
        if 'layer4' in layers:
            activations.append(l4)
        
        return activations
    
    
    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint['memory_bank'] = self.memory_bank

    
    def on_load_checkpoint(self, checkpoint) -> None:
        if 'memory_bank' in checkpoint:
            self.memory_bank = checkpoint['memory_bank']
        else:
            self.memory_bank = torch.tensor([])
            
        if len(self.memory_bank) > self.memory_bank_dim:
            items_to_remove = len(self.memory_bank) - self.memory_bank_dim
            self.memory_bank = self.memory_bank[items_to_remove:]
    
    
    def on_train_epoch_end(self) -> None:
        for current_batch in self.batches_outputs:
            data = current_batch[0]
            y_hat = get_prediction_class(current_batch[1])
            y_true = current_batch[2]
            batch = torch.tensor([])
            for i in range(len(data)):
                if y_hat[i] == 0 and y_true[i] == 0:
                    batch = torch.cat([batch, data[i][None, :]])
            self.memory_bank = torch.cat([self.memory_bank, batch])
        if len(self.memory_bank) > self.memory_bank_dim:
            items_to_remove = len(self.memory_bank) - self.memory_bank_dim
            self.memory_bank = self.memory_bank[items_to_remove:]
        
        
    def forward(self, x:Tensor) -> dict:
        output = {}
        self.activations = {}
        # forwarding through backbone
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        
        
        if 'layer3' in self.activations:
            f3:Tensor = self.activations['layer3']
            f3 = F.adaptive_avg_pool2d(f3, output_size=(1,1))
            f3 = torch.flatten(f3, 1)
            features = torch.cat([f3, features], dim=1)
        if 'layer2' in self.activations:
            f2:Tensor = self.activations['layer2']
            f2 = F.adaptive_avg_pool2d(f2, output_size=(1,1))
            f2 = torch.flatten(f2, 1)
            features = torch.cat([f2, features], dim=1)
        if 'layer1' in self.activations:
            f1:Tensor = self.activations['layer1']
            f1 = F.adaptive_avg_pool2d(f1, output_size=(1,1))
            f1 = torch.flatten(f1, 1)
            features = torch.cat([f1, features], dim=1)
        
        
        # feeding latent space with embedding vector
        embeddings = self.latent_space(features)
        y_hat = self.classifier(embeddings)
        
        output['classifier'] = y_hat
        output['latent_space'] = embeddings
        return output
     
    
    def training_step(self, batch:Tuple[Tensor], batch_idx) -> Tensor:    
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
        current_out = (
            outputs['latent_space'].detach().to('cpu'), 
            y_hat.detach().to('cpu'), 
            y.detach().to('cpu')
            )
        self.batches_outputs.append(current_out)
        return loss

    
    def validation_step(self, batch:Tuple[Tensor], batch_idx) -> dict:
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


    def test_step(self, batch:Tuple[Tensor], batch_idx) -> dict:
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
    
    
    def predict_step(self, batch:Tuple[Tensor], batch_idx, dataloader_idx=0) -> dict:
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
    
    
    def configure_optimizers(self) -> None:
        optimizer = torch.optim.SGD(self.parameters(), self.lr, momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.num_epochs)
        return [optimizer], [scheduler]

 
class GDE():
    def __init__(self) -> None:
        pass
        
    def fit(self, embeddings:Tensor):
        self.kde = KernelDensity().fit(embeddings)
        
        
    def predict(self, embeddings:Tensor):
        scores = self.kde.score_samples(embeddings)
        norm = np.linalg.norm(-scores)
        return torch.tensor(-scores/norm)


class AnomalyDetector:
    def __init__(self) -> None:
        pass
    
    def fit(self, embeddings:Tensor) -> None:
        self.k = 3
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        self.nbrs:NearestNeighbors = NearestNeighbors(
            n_neighbors=self.k, algorithm='auto', metric='cosine').fit(embeddings)


    def predict(self, x:Tensor) -> Tensor:
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        anomaly_scores = self.nbrs.kneighbors(x)[0].squeeze()
        anomaly_scores = torch.tensor(anomaly_scores)
        if self.k > 1:
            anomaly_scores = torch.mean(anomaly_scores, dim=1)
        return anomaly_scores
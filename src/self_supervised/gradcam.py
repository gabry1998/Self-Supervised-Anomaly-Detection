import torch
import torch.nn.functional as F
import numpy as np
from self_supervised.model import SSLModel



class GradCam:
    def __init__(self, model:SSLModel) -> None:
        model.eval()
        self.localizer = model
        self.localizer.set_for_localization(True)
        self.gradients = dict()
        self.activations = dict()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None
        target_layer = self.localizer.feature_extractor.layer4
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def compute_gradcam(self, x, class_idx=None):
        b, c, h, w = x.size()
        logit = self.localizer(x)
        self.localizer.zero_grad()
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()
        
        score.backward(retain_graph=False)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
    
        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        #saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map = F.interpolate(saliency_map, size=(h,w), mode='bilinear')
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map
    
    def __call__(self, input, class_idx=None):
        return self.compute_gradcam(input, class_idx)
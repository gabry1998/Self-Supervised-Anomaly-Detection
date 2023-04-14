from __future__ import annotations
from torch import Tensor
import torch



class ModelOutputsContainer:
    def __init__(self) -> None:
        self.original_data:Tensor = None # tensor original image, not normalized
        self.tensor_data:Tensor = None # images normalized tensor or patches normalized tensor
        self.y_true_binary_labels:Tensor = None # true labels 1D tensor
        self.raw_predictions:Tensor = None # raw classification model output
        self.y_hat:Tensor = None # predicted labels 1D tensor
        self.y_true_multiclass_labels:Tensor = None # labels 1D tensor for tsne visualization
        self.ground_truths:Tensor = None # ground thruths maps tensors
        self.anomaly_maps:Tensor = None # anomaly scores (image-level) vector or anomaly maps (patch-level) vector
        self.embedding_vectors:Tensor = None # 2D tensor PeraNet outputs
        
        self.threshold = None
    
    def to_cpu(self):
        self.original_data = self.original_data.to('cpu') if torch.is_tensor(self.original_data) > 0 else None
        self.tensor_data = self.tensor_data.to('cpu') if torch.is_tensor(self.tensor_data) > 0 else None
        self.y_true_binary_labels = self.y_true_binary_labels.to('cpu') if torch.is_tensor(self.y_true_binary_labels) > 0 else None
        self.raw_predictions = self.raw_predictions.to('cpu') if torch.is_tensor(self.raw_predictions) > 0 else None
        self.y_hat = self.y_hat.to('cpu') if torch.is_tensor(self.y_hat) > 0 else None
        self.y_true_multiclass_labels = self.y_true_multiclass_labels.to('cpu') if torch.is_tensor(self.y_true_multiclass_labels) > 0 else None
        self.ground_truths = self.ground_truths.to('cpu') if torch.is_tensor(self.ground_truths) > 0 else None
        self.anomaly_maps = self.anomaly_maps.to('cpu') if torch.is_tensor(self.anomaly_maps) > 0 else None
        self.embedding_vectors = self.embedding_vectors.to('cpu') if torch.is_tensor(self.embedding_vectors) > 0 else None
    
    def from_list(self, predictions: list[ModelOutputsContainer]):
        a,b,c,d,e,f,g,h,i = [],[],[],[],[],[],[],[],[]

        for p in predictions:
            p.to_cpu()
            a.append(p.original_data)
            b.append(p.tensor_data)
            c.append(p.y_true_binary_labels)
            d.append(p.raw_predictions)
            e.append(p.y_hat)
            f.append(p.y_true_multiclass_labels)
            g.append(p.ground_truths) if torch.is_tensor(p.ground_truths) else []
            h.append(p.anomaly_maps) if torch.is_tensor(p.anomaly_maps) else []
            i.append(p.embedding_vectors)
            
        self.original_data = torch.cat(a) if len(a) else None
        self.tensor_data = torch.cat(b) if len(b) else None
        self.y_true_binary_labels = torch.cat(c) if len(c) else None
        self.raw_predictions = torch.cat(d) if len(d) else None
        self.y_hat = torch.cat(e) if len(e) else None
        self.y_true_multiclass_labels = torch.cat(f) if len(f) else None
        self.ground_truths = torch.cat(g) if len(g) else None
        self.anomaly_maps = torch.cat(h) if len(h) else None
        self.embedding_vectors = torch.cat(i) if len(i) else None
        
        self.threshold = None


class EvaluationOutputContainer:
    def __init__(self, subject:str) -> None:
        self.subject = subject
        
        self.auroc:float = None
        self.f1_score:float = None
        self.aupro:float = None
        self.iou:float = None
        
        self.auroc_fpr_tpr:tuple = None
        self.aupro_fpr_tpr:tuple = None
    
    
    def to_string(self) -> str:
        return self.subject.upper()+" scores: [\n\
    auroc: {0},\n\
    f1-score: {1},\n\
    aupro: {2},\n\
    iou: {3}\n\
]".format(
            round(self.auroc, 2) if self.auroc else None, 
            round(self.f1_score, 2) if self.f1_score else None,
            round(self.aupro, 2) if self.aupro else None,
            round(self.iou, 2) if self.iou else None
        )


class MultipleEvaluationsOuputsContainer:
    def __init__(self) -> None:
        self.subject_list = []
        
        self.auroc_list = []
        self.f1_score_list = []
        self.aupro_list = []
        self.iou_list = []
        
        self.auroc_fpr_tpr_list = []
        self.aupro_fpr_tpr_list = []
    
    def add(self, output_container:EvaluationOutputContainer):
        self.subject_list.append(output_container.subject)
        
        self.auroc_list.append(output_container.auroc) if output_container.auroc else []
        self.f1_score_list.append(output_container.f1_score) if output_container.f1_score else []
        self.aupro_list.append(output_container.aupro) if output_container.aupro else []
        self.iou_list.append(output_container.iou) if output_container.iou else []
        
        self.auroc_fpr_tpr_list.append(output_container.auroc_fpr_tpr) if output_container.auroc_fpr_tpr else []
        self.aupro_fpr_tpr_list.append(output_container.aupro_fpr_tpr) if output_container.aupro_fpr_tpr else []
        
        
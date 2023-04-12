import numpy as np



class ModelOutputsContainer:
    original_data = None # full original image
    tensor_data = None # images normalized tensor or patches normalized tensor
    y_true = None # true labels 1D tensor
    y_hat = None # pred labels 1D tensor
    y_tsne = None # labels 1D tensor for tsne visualization
    ground_truths = None # mvtec ground thruths maps tensors
    anomaly_maps = None # predicted anomaly maps tensors (upsampled)
    embedding_vectors = None # 2D tensor PeraNet output


class EvaluationOutputContainer:
    image_auroc = 0
    classification_auroc = 0
    aupro = 0


def TEXTURES() -> np.ndarray:
    return np.array(['carpet','grid','leather','tile','wood'])


def NON_FIXED_OBJECTS() -> np.ndarray:
    return np.array(['hazelnut', 'screw', 'metal_nut'])

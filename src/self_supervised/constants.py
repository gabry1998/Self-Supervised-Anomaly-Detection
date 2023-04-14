def LOCALIZATION_OUTPUTS() -> list:
    return ['original', 'ground_truth', 'anomaly_map', 'anomaly_map_upsampled', 'localization', 'segmentation']


def METRICS() -> list:
    return ['auroc','f1-score','aupro','iou']


def TEXTURES() -> list:
    return ['carpet','grid','leather','tile','wood']


def OBJECTS() -> list:
    return [
        'bottle',
        'cable',
        'capsule',
        'hazelnut',
        'metal_nut',
        'pill',
        'screw',
        'tile',
        'toothbrush',
        'transistor',
        'zipper'
    ]


def OBJECTS_SET_ONE()->  list:
    return [
        'bottle',
        'cable',
        'capsule',
        'hazelnut',
        'metal_nut']


def OBJECTS_SET_TWO() -> list:
    return [
        'pill',
        'screw',
        'toothbrush',
        'transistor',
        'zipper']


def NON_FIXED_OBJECTS() -> list:
    return ['hazelnut', 'screw', 'metal_nut']


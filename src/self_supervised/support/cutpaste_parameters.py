from torchvision import transforms
import yaml

class CPP:
    cutpaste_augmentations = {
        'jitter_offset': 0.1,
        'patch':{
            'area_ratio': (0.02, 0.15),
            'aspect_ratio': ((0.3, 1),(1, 3.3))
        },
        'scar':{
            'width': (2,16),
            'thiccness': (10,25)
        }
    }
    offset = cutpaste_augmentations['jitter_offset']
    jitter_transforms = transforms.ColorJitter(
                            brightness = offset,
                            contrast = offset,
                            saturation = offset,
                            hue = offset)

    summary = yaml.dump(cutpaste_augmentations, default_flow_style=False)





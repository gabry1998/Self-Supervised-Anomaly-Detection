from torchvision import transforms
import yaml

class CPP:
    jitter_offset = 0.1
    
    rectangle_area_ratio = (0.2, 0.31)
    rectangle_aspect_ratio = ((0.3, 1),(1, 3.3))
    
    scar_area_ratio = (0.02, 0.03)
    scar_aspect_ratio = ((0.3, 1),(1, 3.3))
    
    cutpaste_augmentations = {
        'jitter_offset': 0.1,
        'patch':{
            'area_ratio': (0.02, 0.05),
            'aspect_ratio': ((0.3, 1),(1, 3.3))
        },
        'scar':{
            'width': (2,16),
            'thiccness': (10,25)
        }
    }
    #jitter_offset = cutpaste_augmentations['jitter_offset']
    jitter_transforms = transforms.ColorJitter(
                            brightness = jitter_offset,
                            contrast = jitter_offset,
                            saturation = jitter_offset,
                            hue = jitter_offset)
    summary = yaml.dump(cutpaste_augmentations, default_flow_style=False)





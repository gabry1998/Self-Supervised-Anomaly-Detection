from torchvision import transforms


augmentation_dict = {
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
offset = augmentation_dict['jitter_offset']
augs = transforms.ColorJitter(brightness = offset,
                              contrast = offset,
                              saturation = offset,
                              hue = offset)
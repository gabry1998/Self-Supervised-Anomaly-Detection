from self_supervised.functional import get_all_subject_experiments
from tqdm import tqdm
import self_supervised.tools as tools
import os
os.system('clear')



experiments = get_all_subject_experiments('dataset/')
pbar = tqdm(range(len(experiments)))
for i in pbar:
    subject = experiments[i]
    pbar.set_description('Running training on subject '+subject.upper())
    
    tools.training(
        dataset_dir='dataset/'+subject+'/',
        outputs_dir='brutta_copia/temp/image_level/computations/'+subject+'/',
        subject=subject,
        imsize=(256,256),
        patch_localization=False,
        batch_size=96,
        projection_training_params=(1, 0.03),
        fine_tune_params=(1, 0.01)
    )
    
    os.system('clear')
    


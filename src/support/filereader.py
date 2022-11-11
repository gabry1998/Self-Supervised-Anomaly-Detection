import numpy as np
import glob

def get_image_filenames(main_path, n_repeat=1):
    x = sorted([f for f in glob.glob(main_path+'*.png', recursive = True)])
    return np.repeat(x, n_repeat)
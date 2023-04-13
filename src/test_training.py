import self_supervised.tools as tools

tools.training(
    dataset_dir='dataset/bottle/',
    outputs_dir='brutta_copia/temp/',
    subject='bottle',
    imsize=(256,256),
    patch_localization=False,
    seed=0,
    batch_size=96,
    projection_training_params=(1,0.03), # (epochs, learning_rate)
    fine_tune_params=(1,0.03)
)
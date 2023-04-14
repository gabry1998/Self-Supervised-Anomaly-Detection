from self_supervised.constants import ModelOutputsContainer
from self_supervised.constants import LOCALIZATION_OUTPUTS
import self_supervised.tools as tools



output:ModelOutputsContainer = tools.inference(
    model_input_dir='outputs/patch_level/computations/carpet/best_model.ckpt',
    dataset_dir='dataset/carpet/',
    subject='carpet',
    mvtec_inference=True,
    patch_localization=True,
    max_images_to_inference=10
)

localizer = tools.Localizer(
    outputs_container=output,
    outputs_dir='brutta_copia/temp/localization/',
    outputs_list=LOCALIZATION_OUTPUTS()
)
localizer.localize(
    threshold=output.threshold
)
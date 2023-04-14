from self_supervised.constants import TEXTURES
import self_supervised.tools as tools
from self_supervised.data_containers import \
    ModelOutputsContainer, \
    MultipleEvaluationsOuputsContainer, \
    EvaluationOutputContainer
from self_supervised.functional import get_all_subject_experiments
from tqdm import tqdm
import os
os.system('clear')



experiments = get_all_subject_experiments('dataset/')
pbar = tqdm(range(len(experiments)))

texture_scores_container = MultipleEvaluationsOuputsContainer()
object_scores_container = MultipleEvaluationsOuputsContainer()

for i in pbar:
    subject = experiments[i]
    pbar.set_description('Running evaluation on subject '+subject.upper())
    
    output_artificial = tools.inference(
        model_input_dir='outputs/image_level/computations/'+subject+'/best_model.ckpt',
        dataset_dir='dataset/'+subject+'/',
        subject=subject,
        mvtec_inference=False,
        patch_localization=False
    )
    output_real = tools.inference(
        model_input_dir='outputs/image_level/computations/'+subject+'/best_model.ckpt',
        dataset_dir='dataset/'+subject+'/',
        subject=subject,
        mvtec_inference=True,
        patch_localization=False
    )
    
    evaluator = tools.Evaluator(
        subject=subject,
        evaluation_metrics=['auroc', 'f1-score']
    )
    evaluator.evaluate(
        output_container=output_real,
        outputs_dir='brutta_copia/temp/image_level/plots/'+subject+'/')
    
    if subject in TEXTURES():
        texture_scores_container.add(evaluator.scores)
    else:
        object_scores_container.add(evaluator.scores)
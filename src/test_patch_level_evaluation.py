import self_supervised.tools as tools


output_mvtec = tools.inference(
    model_input_dir='outputs/patch_level/computations/carpet/best_model.ckpt',
    dataset_dir='dataset/carpet/',
    subject='carpet',
    mvtec_inference=True,
    patch_localization=True
)

output_mvtec.anomaly_maps = tools.upsample(output_mvtec.anomaly_maps)
# evaluation
evaluator = tools.Evaluator(
    evaluation_metrics=['auroc','aupro', 'iou']
)
evaluator.evaluate(
    output_container=output_mvtec,
    subject='carpet',
    outputs_dir='brutta_copia/temp/',
    patch_level=True
)

print(evaluator.scores.to_string())
import self_supervised.tools as tools


# inference
output_artificial = tools.inference(
    model_input_dir='outputs/image_level/computations/carpet/best_model.ckpt',
    dataset_dir='dataset/carpet/',
    subject='carpet',
    mvtec_inference=False,
)
output_mvtec = tools.inference(
    model_input_dir='outputs/image_level/computations/carpet/best_model.ckpt',
    dataset_dir='dataset/carpet/',
    subject='carpet',
    mvtec_inference=True,
)


# evaluation
evaluator = tools.Evaluator(
    evaluation_metrics=['auroc','f1-score']
)
evaluator.plot_tsne(
    outputs_artificial=output_artificial,
    outputs_real=output_mvtec,
    subject='carpet',
    outputs_dir='brutta_copia/temp/'
)
evaluator.evaluate(
    output_container=output_mvtec,
    subject='carpet',
    outputs_dir='brutta_copia/temp/'
)

print(evaluator.scores.to_string())
from self_supervised.model import GDE, SSLM
from self_supervised.datasets import *
from tqdm import tqdm
import self_supervised.support.constants as CONST
import self_supervised.support.visualization as vis
import self_supervised.metrics as mtr



def do_inference(model, x):
    if torch.cuda.is_available():
        y_hat, embeddings = model(x.to('cuda'))
    else:
        y_hat, embeddings = model(x.to('cpu'))
    y_hat = y_hat.to('cpu')
    embeddings = embeddings.to('cpu')
    y_hat = get_prediction_class(y_hat)
    
    return y_hat, embeddings


def inference_pipeline(
        dataset_dir:str,
        root_inputs_dir:str,
        root_outputs_dir:str,
        subject:str,
        distortion:bool=False,
        patch_localization=False,
        seed:int=CONST.DEFAULT_SEED(),
        batch_size:int=CONST.DEFAULT_BATCH_SIZE(),
        imsize:int=CONST.DEFAULT_IMSIZE()):
    
    np.random.seed(seed)
    random.seed(seed)
    model_dir = root_inputs_dir+subject+'/image_level/'+'best_model.ckpt'
    print(model_dir)
    outputs_dir = root_outputs_dir+subject+'/image_level/'
    print(outputs_dir)
    sslm = SSLM()
    sslm = SSLM.load_from_checkpoint(model_dir, model=sslm.model)
    sslm.eval()
    if torch.cuda.is_available():
        sslm.to('cuda')
    
    print('')
    print('>>> Generating test dataset (artificial)')
    artificial = GenerativeDatamodule(
        dataset_dir+subject+'/',
        imsize=imsize,
        batch_size=batch_size,
        seed=seed,
        duplication=True,
        min_dataset_length=500,
        patch_localization=patch_localization,
        distortion=distortion
    )
    artificial.setup('test')
    
    print('>>> loading mvtec dataset')
    mvtec = MVTecDatamodule(
                dataset_dir+subject+'/',
                subject=subject,
                imsize=imsize,
                batch_size=batch_size,
                seed=seed
    )
    mvtec.setup()
    
    print('>>> Inferencing...')
    x_artificial, y_artificial = next(iter(artificial.test_dataloader())) 
    y_hat_artificial, embeddings_artificial = do_inference(
        sslm, 
        x_artificial)
    
    print('>>> Printing report')
    df = mtr.report(y=y_artificial, y_hat=y_hat_artificial)
    mtr.export_dataframe(df, saving_path=outputs_dir)

    print('>>> Inferencing over real mvtec images...')
    x_mvtec, gt_mvtec = next(iter(mvtec.test_dataloader())) 
    y_mvtec = gt2label(gt_mvtec, negative=0, positive=3)
    y_hat_mvtec, embeddings_mvtec = do_inference(sslm, x_mvtec)
    
    y_artificial = y_artificial.tolist() 
    total_y = y_artificial + y_mvtec
    total_y = torch.tensor(np.array(total_y))
    total_embeddings = torch.cat([embeddings_artificial, embeddings_mvtec])

    print('>>> Generating tsne visualization')
    start = time.time()
    vis.plot_tsne(
        total_embeddings, 
        total_y, 
        saving_path=outputs_dir, 
        title='Embeddings projection for '+subject.upper())
    end = time.time() - start
    print('Done in '+str(end)+ 'sec')
    
    print('>>> train embeddings for GDE..')
    x_train_mvtec, _ = next(iter(mvtec.train_dataloader())) 
    _, train_embeddings_gde = do_inference(sslm, x_train_mvtec)
    
    embeddings_mvtec = embeddings_mvtec.to('cpu').detach()
    train_embeddings_gde = train_embeddings_gde.to('cpu').detach()
    test_embeddings_gde = torch.nn.functional.normalize(embeddings_mvtec, p=2, dim=1)
    train_embeddings_gde = torch.nn.functional.normalize(train_embeddings_gde, p=2, dim=1)
    
    gde = GDE()
    gde.fit(train_embeddings_gde)
    mvtec_test_scores = gde.predict(test_embeddings_gde)
    mvtec_test_labels = gt2label(gt_mvtec)
    
    print('>>> calculating ROC, AUC, F1..')
    mvtec_test_scores = normalize(mvtec_test_scores)
    fpr, tpr, _ = mtr.compute_roc(mvtec_test_labels, mvtec_test_scores)
    auc_score = mtr.compute_auc(fpr, tpr)
    test_y_hat = multiclass2binary(y_hat_mvtec)
    f_score = mtr.compute_f1(torch.tensor(mvtec_test_labels), test_y_hat)
    
    
    print('>>> plot ROC..')
    vis.plot_roc(
        fpr, tpr, 
        auc_score, 
        saving_path=outputs_dir,
        title='Roc curve for '+subject.upper())
    
    return auc_score, f_score

if __name__ == "__main__":
    root_outputs_dir='brutta_copia/computations/'
    
    experiments = get_all_subject_experiments('dataset/', patch_localization=False)
    
    pbar = tqdm(range(len(experiments)))
    metric_dict = {}
    
    auc_scores_img_lvl = []
    f1_scores_img_lvl = []
    for i in pbar:
        pbar.set_description('Pipeline Execution image level | current subject is '+experiments[i][0].upper())
        subject = experiments[i][0]
        patch_localization = experiments[i][1]
        
        auc_score, f_score = inference_pipeline(
            dataset_dir='dataset/', 
            root_inputs_dir='brutta_copia/computations/',
            root_outputs_dir=root_outputs_dir,
            subject=subject,
            patch_localization=patch_localization,
            distortion=False,
            batch_size=128,
            imsize=(256,256))
        auc_scores_img_lvl.append(auc_score)
        f1_scores_img_lvl.append(f_score)
        os.system('clear')
    metric_dict['auc (image level)'] = np.array(auc_scores_img_lvl)
    metric_dict['f1 (image level)'] = np.array(f1_scores_img_lvl)

    report = mtr.metrics_to_dataframe(metric_dict, np.array([x[0] for x in experiments]))
    mtr.export_dataframe(report, saving_path=root_outputs_dir, name='roc_and_f1_scores.csv')

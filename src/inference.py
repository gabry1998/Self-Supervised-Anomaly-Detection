from self_supervised.model import GDE, SSLM
from self_supervised.datasets import *
from tqdm import tqdm
import self_supervised.support.constants as CONST
import self_supervised.support.visualization as vis
import self_supervised.metrics as mtr

def inference_image_level_pipeline(
        dataset_dir:str,
        root_inputs_dir:str,
        root_outputs_dir:str,
        subject:str,
        patch_type:str='standard',
        seed:int=CONST.DEFAULT_SEED(),
        batch_size:int=CONST.DEFAULT_BATCH_SIZE(),
        imsize:int=CONST.DEFAULT_IMSIZE()):
    
    if patch_type=='deformed':
        distortion=True
    else:
        distortion=False
    
    if torch.cuda.is_available():
        cuda_available = True
    else:
        cuda_available = False
    
    model_dir = root_inputs_dir+subject+'/image_level/'+'best_model.ckpt'
    print(model_dir)
    outputs_dir = root_outputs_dir+subject+'/image_level/'
    print(outputs_dir)
    sslm = SSLM()
    sslm = SSLM.load_from_checkpoint(model_dir, model=sslm.model)
    sslm.eval()
    if cuda_available:
        sslm.to('cuda')
    
    print('')
    print('>>> Generating test dataset (artificial)')
    start = time.time()
    artificial = GenerativeDatamodule(
        dataset_dir+subject+'/',
        imsize=imsize,
        batch_size=batch_size,
        seed=seed,
        duplication=True,
        min_dataset_length=500,
        patch_localization=False,
        distortion=distortion
    )
    artificial.setup('test')
    end = time.time() - start
    print('Generated in '+str(end)+ 'sec')
    
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
    start = time.time()
    x_artificial, y_artificial = next(iter(artificial.test_dataloader())) 
    if cuda_available:
        y_hat_artificial, embeddings_artificial = sslm(x_artificial.to('cuda'))
    else:
        y_hat_artificial, embeddings_artificial = sslm(x_artificial.to('cpu'))
    y_hat_artificial = y_hat_artificial.to('cpu')
    embeddings_artificial = embeddings_artificial.to('cpu')
    y_hat_artificial = get_prediction_class(y_hat_artificial)
    end = time.time() - start
    print('Done in '+str(end)+ 'sec')
    
    print('>>> Printing report')
    df = mtr.report(y=y_artificial, y_hat=y_hat_artificial)
    #df.to_csv(outputs_dir+'metric_report.csv', index = False)
    mtr.export_dataframe(df, saving_path=outputs_dir)

    print('>>> Inferencing over real mvtec images...')
    start = time.time()
    x_mvtec, gt_mvtec = next(iter(mvtec.test_dataloader())) 
    y_mvtec = gt2label(gt_mvtec, negative=0, positive=3)
           
    if cuda_available:
        y_hat_mvtec, embeddings_mvtec = sslm(x_mvtec.to('cuda'))
    else:
        y_hat_mvtec, embeddings_mvtec = sslm(x_mvtec.to('cpu'))
    y_hat_mvtec = y_hat_mvtec.to('cpu')
    embeddings_mvtec = embeddings_mvtec.to('cpu')  
    y_hat_mvtec = get_prediction_class(y_hat_mvtec)
    end = time.time() - start
    print('Done in '+str(end)+ 'sec')
    
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
    
    print('>>> calculating ROC curve..')
    start = time.time()
    train_embeddings_gde = []
    for x, _ in mvtec.train_dataloader():
        _, batch_train_embeddings = sslm(x.to('cuda'))
        train_embeddings_gde.append(batch_train_embeddings.to('cpu'))
    train_embeddings_gde = torch.cat(train_embeddings_gde).to('cpu').detach()
    
    gt_mvtec_test = []
    test_embeddings_gde = []
    test_y_hat = []
    with torch.no_grad():
        for x, label in mvtec.test_dataloader():
            y_hat_mvtec, batch_embeddings = sslm(x.to('cuda'))
            y_hat_mvtec = y_hat_mvtec.to('cpu').detach()
            
            test_embeddings_gde.append(batch_embeddings.to('cpu').detach())
            test_y_hat.append(get_prediction_class(y_hat_mvtec))
            gt_mvtec_test.append(label.to('cpu').detach())
            
    gt_mvtec_test = torch.cat(gt_mvtec_test)
    test_embeddings_gde = torch.cat(test_embeddings_gde)
    test_y_hat = torch.cat(test_y_hat)
    
    test_embeddings_gde = torch.nn.functional.normalize(test_embeddings_gde, p=2, dim=1)
    train_embeddings_gde = torch.nn.functional.normalize(train_embeddings_gde, p=2, dim=1)
    
    gde = GDE()
    gde.fit(train_embeddings_gde)
    mvtec_test_scores = gde.predict(test_embeddings_gde)
    mvtec_test_labels = gt2label(gt_mvtec_test)
    
    vis.plot_roc(
        mvtec_test_labels, 
        mvtec_test_scores, 
        saving_path=outputs_dir,
        title='Roc curve for '+subject.upper())
    end = time.time() - start
    print('Done in '+str(end)+ 'sec')
    
    print('>>> AUC and F1')
    mvtec_test_scores = normalize(mvtec_test_scores)
    fpr, tpr, _ = mtr.compute_roc(mvtec_test_labels, mvtec_test_scores)
    auc_score = mtr.compute_auc(fpr, tpr)
    test_y_hat = multiclass2binary(test_y_hat)
    f_score = mtr.compute_f1(torch.tensor(mvtec_test_labels), test_y_hat)
    
    return auc_score, f_score

if __name__ == "__main__":
    root_outputs_dir='outputs/computations/'
    
    experiments = np.array([
        'bottle',
        'grid',
        'screw',
        'tile',
        'toothbrush'
    ])
    
    auc_scores = []
    f1_scores = []
    pbar = tqdm(range(len(experiments)))
    for i in pbar:
        pbar.set_description('Pipeline Execution | current subject is '+experiments[i].upper())
        auc_score, f_score = inference_image_level_pipeline(
            dataset_dir='dataset/', 
            root_inputs_dir='outputs/computations/',
            root_outputs_dir=root_outputs_dir,
            subject=experiments[i],
            batch_size=128,
            imsize=(256,256))
        auc_scores.append(auc_score)
        f1_scores.append(f_score)
        os.system('clear')
        
    metric_dict = {
        'auc (image level)':np.array(auc_scores),
        'f1 (image level)':np.array(f1_scores)
    }
    report = mtr.metrics_to_dataframe(metric_dict, experiments)
    mtr.export_dataframe(report, saving_path=root_outputs_dir, name='roc_and_f1_scores.csv')

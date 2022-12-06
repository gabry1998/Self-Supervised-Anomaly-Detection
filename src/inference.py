from self_supervised.gradcam import GradCam
from self_supervised.model import GDE, SSLM
from self_supervised.datasets import *
from tqdm import tqdm
import self_supervised.datasets as dt
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
        imsize:int=CONST.DEFAULT_IMSIZE(),
        current_auc:int=0,
        current_f1:int=0,
        current_aupro:int=0):
    
    np.random.seed(seed)
    random.seed(seed)
    print('root input directory:', root_inputs_dir)
    model_dir = root_inputs_dir+subject+'/image_level/'+'best_model.ckpt'
    print('model weights dir:', model_dir)
    outputs_dir = root_outputs_dir+subject+'/image_level/'
    print('outputs directory:', outputs_dir)
    print('distorted patches:', distortion)
    print('patch localization:', patch_localization)
    
    print('')
    print('>>> Loading model')
    sslm = SSLM()
    sslm = SSLM.load_from_checkpoint(model_dir, model=sslm.model)
    sslm.eval()
    if torch.cuda.is_available():
        sslm.to('cuda')
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
    start = time.time()
    y_hat_artificial, embeddings_artificial = do_inference(
        sslm, 
        x_artificial)
    end = time.time() - start
    print('Done in '+str(end)+ 'sec')
    
    #print('>>> Printing report')
    #df = mtr.report(y=y_artificial, y_hat=y_hat_artificial)
    #mtr.export_dataframe(df, saving_path=outputs_dir)

    print('>>> Inferencing over real mvtec images...')
    x_mvtec, gt_mvtec = next(iter(mvtec.test_dataloader()))  
    y_mvtec = gt2label(gt_mvtec, negative=0, positive=3)
    start = time.time()
    y_hat_mvtec, embeddings_mvtec = do_inference(sslm, x_mvtec)
    end = time.time() - start
    print('Done in '+str(end)+ 'sec')
    
    print('>>> Embeddings for GDE..')
    start = time.time()
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
    end = time.time() - start
    print('Done in '+str(end)+ 'sec')
    
    print('>>> calculating ROC, AUC, F1..')
    start = time.time()
    mvtec_test_scores = normalize(mvtec_test_scores)
    fpr, tpr, _ = mtr.compute_roc(mvtec_test_labels, mvtec_test_scores)
    auc_score = mtr.compute_auc(fpr, tpr)
    test_y_hat = multiclass2binary(y_hat_mvtec)
    f_score = mtr.compute_f1(torch.tensor(mvtec_test_labels), test_y_hat)
    end = time.time() - start
    print('Done in '+str(end)+ 'sec')
    
    print('>>> plot ROC..')
    if auc_score > current_auc:
        vis.plot_curve(
            fpr, tpr, 
            auc_score, 
            saving_path=outputs_dir,
            title='Roc curve for '+subject.upper(),
            name='roc.png')
    
    print('>>> compute PRO')
    gradcam = GradCam(
        SSLM.load_from_checkpoint(model_dir).model)
    ground_truth_maps = []
    anomaly_maps = []
    for i in range(len(test_y_hat)):
        predicted_class = test_y_hat[i]
        if predicted_class == 0:
            saliency_map = torch.zeros((256,256))[None, :]
        else:
            if predicted_class > 1:
                predicted_class = 1
            x = x_mvtec[i]
            saliency_map = gradcam(x[None, :], test_y_hat[i])
        anomaly_maps.append(np.array(saliency_map.squeeze()))
    anomaly_maps = np.array(anomaly_maps)
    ground_truth_maps = np.array(gt_mvtec.squeeze())
    all_fprs, all_pros = mtr.compute_pro(
    anomaly_maps=anomaly_maps,
    ground_truth_maps=ground_truth_maps)

    au_pro = mtr.compute_aupro(all_fprs, all_pros, 0.3)
    if au_pro > current_aupro:
        vis.plot_curve(
            all_fprs,
            all_pros,
            au_pro,
            saving_path=root_outputs_dir+subject+'/image_level/',
            title='Pro curve for '+subject.upper(),
            name='pro.png'
        )
    
    y_artificial = y_artificial.tolist() 
    total_y = y_artificial + y_mvtec
    total_y = torch.tensor(np.array(total_y))
    total_embeddings = torch.cat([embeddings_artificial, embeddings_mvtec])
    
    print('>>> Generating tsne visualization')
    if f_score > current_f1:
        start = time.time()
        vis.plot_tsne(
            total_embeddings, 
            total_y, 
            saving_path=outputs_dir, 
            title='Embeddings projection for '+subject.upper())
        end = time.time() - start
        print('Done in '+str(end)+ 'sec')
    return auc_score, f_score, au_pro


def run(
        experiments_list:list,
        dataset_dir:str,
        root_inputs_dir:str,
        root_outputs_dir:str,
        num_experiments_for_each_subject:int=1,
        seed_list:list=[0],
        distortion:bool=False,
        patch_localization=False,
        batch_size:int=128,
        imsize:int=CONST.DEFAULT_IMSIZE()):
    
    os.system('clear')
    assert(len(seed_list) == num_experiments_for_each_subject)
    
    pbar = tqdm(range(len(experiments_list)))
    
    metric_dict = {} 
    auc_scores = []
    f1_scores= []
    aupro_scores = []
    auc_score = 0
    aupro_score = 0
    for i in pbar:
        subject = experiments_list[i]
        temp_auc = []
        temp_f1 = []
        temp_aupro = []
        for j in range(num_experiments_for_each_subject):
            seed = seed_list[j]
            pbar.set_description('Inference pipeline | current subject is '+experiments_list[i].upper())
            print('')
            print('Running experiment '+str(j+1)+'/'+str(num_experiments_for_each_subject))
            print('Experiment seed:', str(seed))
            auc_score, f_score, aupro_score = inference_pipeline(
                dataset_dir=dataset_dir,
                root_inputs_dir=root_inputs_dir,
                root_outputs_dir=root_outputs_dir,
                subject=subject,
                distortion=distortion,
                patch_localization=patch_localization,
                seed=seed,
                batch_size=batch_size,
                imsize=imsize,
                current_auc=auc_score,
                current_aupro=aupro_score)
            temp_auc.append(auc_score)
            temp_f1.append(f_score)
            temp_aupro.append(aupro_score)
            os.system('clear')
        
        temp_auc = np.array(temp_auc)
        temp_f1 = np.array(temp_f1)
        temp_aupro = np.array(temp_aupro)
        auc_scores.append(np.mean(temp_auc))
        f1_scores.append(np.mean(temp_f1))
        aupro_scores.append(np.mean(temp_aupro))
        
        
    experiments_list.append('average')
    metric_dict['AUC (image level)'] = np.append(
        auc_scores, 
        np.mean(auc_scores))
    metric_dict['F1 (image level)'] = np.append(
        f1_scores, 
        np.mean(f1_scores))
    metric_dict['AUPRO (image level)'] = np.append(
        aupro_scores, 
        np.mean(aupro_scores))
    
    report = mtr.metrics_to_dataframe(metric_dict, np.array(experiments_list))
    mtr.export_dataframe(report, saving_path=root_outputs_dir, name='image_level_scores.csv')
    
if __name__ == "__main__":
    
    experiments = get_all_subject_experiments('dataset/')
    run(
        experiments_list=experiments,
        dataset_dir='dataset/',
        root_inputs_dir='outputs/computations/',
        root_outputs_dir='brutta_copia/',
        num_experiments_for_each_subject=5,
        seed_list=[0, 1, 2, 3, 4],
        distortion=False,
        patch_localization=False,
        batch_size=32,
        imsize=(256,256))

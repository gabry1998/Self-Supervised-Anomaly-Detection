from self_supervised.gradcam import GradCam
from self_supervised.model import GDE, SSLM
from self_supervised.datasets import *
from tqdm import tqdm
import self_supervised.datasets as dt
import self_supervised.support.constants as CONST
import self_supervised.support.visualization as vis
import self_supervised.metrics as mtr
import time
import random
import numpy as np

class Tracker:
    def __init__(self) -> None:
        self.auc = -1
        self.aupro = -1
        self.auc_pixel = -1


def do_inference(model, x):
    if torch.cuda.is_available():
        with torch.no_grad():
            y_hat, embeddings = model(x.to('cuda'))
    else:
        with torch.no_grad():
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
        polygoned:bool=False,
        colorized_scar:bool=False,
        patch_localization=False,
        seed:int=CONST.DEFAULT_SEED(),
        batch_size:int=CONST.DEFAULT_BATCH_SIZE(),
        imsize:int=CONST.DEFAULT_IMSIZE(),
        tracker:Tracker=None):
    
    np.random.seed(seed)
    random.seed(seed)
    print('root input directory:', root_inputs_dir)
    model_dir = root_inputs_dir+subject+'/image_level/'+'best_model.ckpt'
    print('model weights dir:', model_dir)
    outputs_dir = root_outputs_dir+subject+'/image_level/'
    print('outputs directory:', outputs_dir)
    print('polygoned patches:', polygoned)
    print('patch localization:', patch_localization)
    
    print('')
    print('>>> Loading model')
    sslm = SSLM(dims=[512,512,512,512,512,512,512,512,512])
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
        polygoned=polygoned,
        colorized_scar=colorized_scar,
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
    
    tester = pl.Trainer(
        precision=16,
        #benchmark=True,
        #deterministic=True,
        accelerator='auto', 
        devices=1)
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
    y_hat_gde, train_embeddings_gde = do_inference(sslm, x_train_mvtec)
    y_hat_gde = multiclass2binary(y_hat_gde)
    embeddings_mvtec = embeddings_mvtec.to('cpu').detach()
    train_embeddings_gde = train_embeddings_gde.to('cpu').detach()
    
    test_embeddings_gde = torch.nn.functional.normalize(embeddings_mvtec, p=2, dim=1)
    train_embeddings_gde = torch.nn.functional.normalize(train_embeddings_gde, p=2, dim=1)
    embeddings_artificial = torch.nn.functional.normalize(embeddings_artificial, p=2, dim=1)
    
    gde = GDE()
    gde.fit(train_embeddings_gde)
    mvtec_test_scores = gde.predict(test_embeddings_gde)
    mvtec_test_labels = gt2label(gt_mvtec)
    end = time.time() - start
    print('Done in '+str(end)+ 'sec')
    
    print('>>> calculating (IMAGE LEVEL) ROC, AUC, F1..')
    start = time.time()
    mvtec_test_scores = normalize(mvtec_test_scores)
    fpr, tpr, _ = mtr.compute_roc(mvtec_test_labels, mvtec_test_scores)
    auc_score = mtr.compute_auc(fpr, tpr)
    test_y_hat = multiclass2binary(y_hat_mvtec)
    f_score = mtr.compute_f1(torch.tensor(mvtec_test_labels), test_y_hat)
    end = time.time() - start
    print('Done in '+str(end)+ 'sec')
    
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
    anomaly_maps_np = np.array(anomaly_maps)
    gt_mvtec = gt_mvtec.squeeze()
    ground_truth_maps = np.array(gt_mvtec)
    
    all_fprs, all_pros = mtr.compute_pro(
    anomaly_maps=anomaly_maps_np,
    ground_truth_maps=ground_truth_maps)

    au_pro = mtr.compute_aupro(all_fprs, all_pros, 0.3)
    
    if (auc_score > tracker.auc):
        print('>>> plot ROC..')
        tracker.auc = auc_score
        vis.plot_curve(
            fpr, tpr, 
            auc_score, 
            saving_path=outputs_dir,
            title='Roc curve for '+subject.upper()+' ['+str(seed)+']',
            name='roc.png')
        
        print('>>> Generating tsne visualization')
        y_artificial = y_artificial.tolist() 
        total_y = y_artificial + y_mvtec
        total_y = torch.tensor(np.array(total_y))
        total_embeddings = torch.cat([embeddings_artificial, test_embeddings_gde])
        vis.plot_tsne(
            total_embeddings, 
            total_y, 
            saving_path=outputs_dir, 
            title='Embeddings projection for '+subject.upper()+' ['+str(seed)+']')
    
    if (au_pro > tracker.aupro): 
        print('>>> plot PRO..')
        tracker.aupro = au_pro
        vis.plot_curve(
            all_fprs,
            all_pros,
            au_pro,
            saving_path=root_outputs_dir+subject+'/image_level/',
            title='Pro curve for '+subject.upper()+' ['+str(seed)+']',
            name='pro.png'
        )
    
    print('>>> calculating (PIXEL LEVEL) ROC, AUC, F1..')
    start = time.time()
    flat_anomaly_maps = torch.nan_to_num(torch.tensor(anomaly_maps).flatten(0, -1))
    flat_gt_labels = gt_mvtec.flatten(0, -1)
    fpr, tpr, _ = mtr.compute_roc(flat_gt_labels, flat_anomaly_maps)
    pixel_auc_score = mtr.compute_auc(fpr, tpr)
    pixel_f_score = mtr.compute_f1(flat_gt_labels, torch.tensor(heatmap2mask(flat_anomaly_maps)))
    end = time.time() - start
    print('Done in '+str(end)+ 'sec')
    
    if (pixel_auc_score > tracker.auc_pixel):
        print('>>> plot (PIXEL) ROC..')
        tracker.auc = auc_score
        vis.plot_curve(
            fpr, tpr, 
            pixel_auc_score, 
            saving_path=outputs_dir,
            title='Roc curve for '+subject.upper()+' ['+str(seed)+']',
            name='pixel_roc.png')
        
    return auc_score, f_score, au_pro, pixel_auc_score, pixel_f_score, tracker


def run(
        experiments_list:list,
        dataset_dir:str,
        root_inputs_dir:str,
        root_outputs_dir:str,
        num_experiments_for_each_subject:int=1,
        seed_list:list=[0],
        polygoned:bool=False,
        colorized_scar=False,
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
    pixel_auc_scores = []
    pixel_f1_scores = []
    
    for i in pbar:
        subject = experiments_list[i]
        temp_auc = []
        temp_f1 = []
        temp_aupro = []
        temp_pixel_auc = []
        temp_pixel_f1 = []
        metric_tracker = Tracker()
        for j in range(num_experiments_for_each_subject):
            seed = seed_list[j]
            pbar.set_description('Inference pipeline | current subject is '+experiments_list[i].upper())
            print('')
            print('Running experiment '+str(j+1)+'/'+str(num_experiments_for_each_subject))
            print('Experiment seed:', str(seed))
            auc_score, f_score, aupro_score, pixel_auc_score, pixel_f_score, metric_tracker = inference_pipeline(
                dataset_dir=dataset_dir,
                root_inputs_dir=root_inputs_dir,
                root_outputs_dir=root_outputs_dir,
                subject=subject,
                polygoned=polygoned,
                colorized_scar=colorized_scar,
                patch_localization=patch_localization,
                seed=seed,
                batch_size=batch_size,
                imsize=imsize,
                tracker=metric_tracker)
            temp_auc.append(auc_score)
            temp_f1.append(f_score)
            temp_aupro.append(aupro_score)
            temp_pixel_auc.append(pixel_auc_score)
            temp_pixel_f1.append(pixel_f_score)
            os.system('clear')
        
        temp_auc = np.array(temp_auc)
        temp_f1 = np.array(temp_f1)
        temp_aupro = np.array(temp_aupro)
        auc_scores.append(np.mean(temp_auc))
        f1_scores.append(np.mean(temp_f1))
        aupro_scores.append(np.mean(temp_aupro))
        pixel_auc_scores.append(np.mean(temp_pixel_auc))
        pixel_f1_scores.append(np.mean(temp_pixel_f1))
        
        
    experiments_list.append('average')
    metric_dict['AUC (image)'] = np.append(
        auc_scores, 
        np.mean(auc_scores))
    metric_dict['F1 (image)'] = np.append(
        f1_scores, 
        np.mean(f1_scores))
    metric_dict['AUC (pixel)'] = np.append(
        pixel_auc_scores, 
        np.mean(pixel_auc_scores))
    metric_dict['F1 (pixel)'] = np.append(
        pixel_f1_scores, 
        np.mean(pixel_f1_scores))
    metric_dict['AUPRO'] = np.append(
        aupro_scores, 
        np.mean(aupro_scores))
    
    report = mtr.metrics_to_dataframe(metric_dict, np.array(experiments_list))
    mtr.export_dataframe(report, saving_path=root_outputs_dir, name='polygon_patch_swirl.csv')
    
    
if __name__ == "__main__":
    
    experiments = get_all_subject_experiments('dataset/')
    run(
        experiments_list=experiments,
        dataset_dir='dataset/',
        root_inputs_dir='outputs/computations/',
        root_outputs_dir='brutta_copia/computations/',
        num_experiments_for_each_subject=3,
        seed_list=[
            204110176,
            129995678,
            123456789],
        polygoned=True,
        colorized_scar=False,
        patch_localization=False,
        batch_size=128,
        imsize=(256,256))

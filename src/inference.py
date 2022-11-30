from self_supervised.model import GDE, SSLM
from self_supervised.datasets import *
from tqdm import tqdm
import self_supervised.support.visualization as vis
import self_supervised.metrics as mtr


def inference_pipeline(
        dataset_dir:str,
        root_dir:str,
        subject:str,
        level:str,
        args:dict=None):
    
    if torch.cuda.is_available():
        cuda_available = True
    else:
        cuda_available = False
        
    imsize = args['imsize']
    batch_size = args['batch_size']
    seed = args['seed']
    
    results_dir = root_dir+subject+'/'+level+'/'
    print(results_dir)
    model_dir = results_dir+'best_model.ckpt'
    print(model_dir)
    sslm = SSLM()
    sslm = SSLM.load_from_checkpoint(model_dir, model=sslm.model)
    sslm.eval()
    if cuda_available:
        sslm.to('cuda')
    
    print('')
    print('>>> Generating test dataset (artificial)')
    start = time.time()
    if level == 'image_level':
        artificial = GenerativeDatamodule(
            dataset_dir+subject+'/',
            imsize=imsize,
            batch_size=batch_size,
            seed=seed,
            duplication=True,
            min_dataset_length=500,
            patch_localization=False
        )
    else:
        artificial = GenerativeDatamodule(
            dataset_dir+subject+'/',
            imsize=imsize,
            batch_size=batch_size,
            seed=seed,
            duplication=True,
            min_dataset_length=500,
            patch_localization=True
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
    x_artificial, y_artificial = next(iter(artificial.test_dataloader())) 
    if cuda_available:
        y_hat_artificial, embeddings_artificial = sslm(x_artificial.to('cuda'))
    else:
        y_hat_artificial, embeddings_artificial = sslm(x_artificial.to('cpu'))
    y_hat_artificial = y_hat_artificial.to('cpu')
    embeddings_artificial = embeddings_artificial.to('cpu')
    y_hat_artificial = get_prediction_class(y_hat_artificial)
    
    print('>>> Printing report')
    df = mtr.report(y=y_artificial, y_hat=y_hat_artificial)
    df.to_csv(results_dir+'/metric_report.csv', index = False)

    print('>>> Inferencing over real mvtec images...')
    x_mvtec, gt_mvtec = next(iter(mvtec.test_dataloader())) 
    y_mvtec = gt2label(gt_mvtec, negative=0, positive=3)
           
    if cuda_available:
        y_hat_mvtec, embeddings_mvtec = sslm(x_mvtec.to('cuda'))
    else:
        y_hat_mvtec, embeddings_mvtec = sslm(x_mvtec.to('cpu'))
    y_hat_mvtec = y_hat_mvtec.to('cpu')
    embeddings_mvtec = embeddings_mvtec.to('cpu')  
    y_hat_mvtec = get_prediction_class(y_hat_mvtec)
    
    y_artificial = y_artificial.tolist() 
    total_y = y_artificial + y_mvtec
    total_y = torch.tensor(np.array(total_y))
    
    total_embeddings = torch.cat([embeddings_artificial, embeddings_mvtec])

    print('>>> Generating tsne visualization')
    vis.plot_tsne(total_embeddings, total_y, results_dir, subject)
    
    print('>>> calculating ROC AUC curve..')
    train_embeddings_gde = []
    for x, _ in mvtec.train_dataloader():
        _, batch_train_embeddings = sslm(x.to('cuda'))
        train_embeddings_gde.append(batch_train_embeddings.to('cpu'))
    train_embeddings_gde = torch.cat(train_embeddings_gde).to('cpu').detach()
    
    gt_mvtec_test = []
    test_embeddings_gde = []
    with torch.no_grad():
        for x, label in mvtec.test_dataloader():
            _, batch_embeddings = sslm(x.to('cuda'))
            test_embeddings_gde.append(batch_embeddings.to('cpu').detach())
            gt_mvtec_test.append(label.to('cpu').detach())
    gt_mvtec_test = torch.cat(gt_mvtec_test)
    test_embeddings_gde = torch.cat(test_embeddings_gde)
    
    test_embeddings_gde = torch.nn.functional.normalize(test_embeddings_gde, p=2, dim=1)
    train_embeddings_gde = torch.nn.functional.normalize(train_embeddings_gde, p=2, dim=1)
    
    gde = GDE()
    gde.fit(train_embeddings_gde)
    mvtec_test_scores = gde.predict(test_embeddings_gde)
    mvtec_test_labels = gt2label(gt_mvtec_test)
    
    vis.plot_roc(mvtec_test_labels, mvtec_test_scores, subject, results_dir+'/roc.png')

if __name__ == "__main__":
    dataset_dir = 'dataset/'
    root_dir = 'outputs/computations/' 
    imsize=(256,256)
    batch_size = 128
    seed = 0
    
    args = {
        'imsize': imsize,
        'batch_size': batch_size,
        'seed': seed,
    }
    
    experiments = [
        ('grid', 'image_level')
    ]
    
    pbar = tqdm(range(len(experiments)))
    for i in pbar:
        pbar.set_description('Pipeline Execution | current subject is '+experiments[i][0].upper())
        inference_pipeline(
            dataset_dir, 
            root_dir, 
            experiments[i][0],
            experiments[i][1],
            args)
        os.system('clear')

from self_supervised.model import GDE, SSLM, SSLModel, MetricTracker
from self_supervised.datasets import *
import pytorch_lightning as pl
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import self_supervised.support.constants as CONST
from self_supervised.support.visualization import plot_roc


def inference_pipeline(
        root_dir:str,
        outputs_dir:str,
        subject:str,
        args:dict=None):
    
    if torch.cuda.is_available():
        cuda_available = True
    else:
        cuda_available = False
        
    imsize = args['imsize']
    batch_size = args['batch_size']
    seed = args['seed']
    
    results_dir = outputs_dir+subject
    model_dir = results_dir+'/best_model.ckpt'
    dataset_dir = root_dir+subject+'/'
    sslm = SSLM()
    sslm = SSLM.load_from_checkpoint(model_dir, model=sslm.model)
    sslm.eval()
    if cuda_available:
        sslm.to('cuda')
    
    print('')
    print('>>> Generating test dataset (artificial)')
    start = time.time()
    datamodule = GenerativeDatamodule(
                dataset_dir,
                imsize=imsize,
                batch_size=batch_size,
                seed=seed,
                min_dataset_length=500,
                duplication=True
    )
    datamodule.setup('test')
    end = time.time() - start
    print('Generated in '+str(end)+ 'sec')
    
    print('>>> loading mvtec dataset')
    mvtec = MVTecDatamodule(
                dataset_dir,
                subject=subject,
                imsize=imsize,
                batch_size=batch_size,
                seed=seed
    )
    mvtec.setup()
    print('>>> Inferencing...')
    x, y = next(iter(datamodule.test_dataloader())) 
    if cuda_available:
        x = x.to('cuda')
        y_hat, embeddings = sslm(x)
        y_hat = y_hat.to('cpu')
        embeddings = embeddings.to('cpu')
    else:
        x = x.to('cpu')
        y_hat, embeddings = sslm(x)
        
    y_hat = torch.max(y_hat.data, 1)
    y_hat = y_hat.indices
    
    print('>>> Printing report')
    
    result = classification_report( 
            y,
            y_hat,
            labels=[0,1,2],
            output_dict=True
        )
    df = pd.DataFrame.from_dict(result)
    df.to_csv(results_dir+'/metric_report.csv', index = False)
    
    print('>>> Inferencing over real mvtec images...')
    x, gt = next(iter(mvtec.test_dataloader())) 
    mvtec_labels = []
    for ground_truth in gt:
        if torch.sum(ground_truth) == 0:
            mvtec_labels.append(0)
        else:
            mvtec_labels.append(3)
            
    if cuda_available:
        x = x.to('cuda')
        y_hat_mvtec, embeddings_mvtec = sslm(x)
        y_hat_mvtec = y_hat_mvtec.to('cpu')
        embeddings_mvtec = embeddings_mvtec.to('cpu')
    else:
        x = x.to('cpu')
        y_hat_mvtec, embeddings_mvtec = sslm(x)
        
    y_hat_mvtec = torch.max(y_hat_mvtec.data, 1)
    y_hat_mvtec = y_hat_mvtec.indices
    
    y_artificial = y.tolist()
    
    total_y = y_artificial + mvtec_labels
    total_y = torch.tensor(np.array(total_y))
    
    tot_embeddings = torch.cat([embeddings, embeddings_mvtec])

    print('>>> Generating tsne visualization')
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(tot_embeddings.detach().numpy())
    tx = tsne_results[:, 0]
    ty = tsne_results[:, 1]

    df = pd.DataFrame()
    df["labels"] = total_y
    df["comp-1"] = tx
    df["comp-2"] = ty
    plt.figure()

    sns.scatterplot(hue=df.labels.tolist(),
                    x='comp-1',
                    y='comp-2',
                    palette=sns.color_palette("hls", 4),
                    data=df).set(title='Embeddings projection ('+subject+')')
    plt.savefig(results_dir+'/tsne.png')
    
    print('>>> calculating ROC AUC curve..')
    train_embed = []
    for x, _ in mvtec.train_dataloader():
        _, train_embeddings = sslm(x.to('cuda'))
        train_embeddings = train_embeddings.to('cpu')
        train_embed.append(train_embeddings)
    train_embed = torch.cat(train_embed).to('cpu').detach()
    
    test_labels = []
    test_embeds = []
    with torch.no_grad():
        for x, label in mvtec.test_dataloader():
            y_hat, embeddings = sslm(x.to('cuda'))

            # save 
            test_embeds.append(embeddings.to('cpu').detach())
            test_labels.append(label.to('cpu').detach())
    test_labels = torch.cat(test_labels)
    test_embeds = torch.cat(test_embeds)
    
    test_embeds = torch.nn.functional.normalize(test_embeds, p=2, dim=1)
    train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)
    
    gde = GDE()
    gde.fit(train_embed)
    scores = gde.predict(test_embeds)

    int_labels = []
    for x in test_labels:
      if torch.sum(x) == 0:
        int_labels.append(0)
      else:
        int_labels.append(1)
    test_labels = torch.tensor(int_labels)
    
    plot_roc(test_labels, scores, subject, results_dir+'/roc.png')

if __name__ == "__main__":
    dataset_dir = 'dataset/'
    results_dir = 'outputs/computations/' 
    imsize=(256,256)
    batch_size = 128
    seed = 0
    
    args = {
        'imsize': imsize,
        'batch_size': batch_size,
        'seed': seed,
    }
    
    experiments = [
        'screw'
    ]
    
    pbar = tqdm(range(len(experiments)))
    for i in pbar:
        pbar.set_description('Pipeline Execution | current subject is '+experiments[i].upper())
        inference_pipeline(
            dataset_dir, 
            results_dir, 
            experiments[i],
            args)
        os.system('clear')

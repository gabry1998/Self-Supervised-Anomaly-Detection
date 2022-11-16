from self_supervised.model import SSLM, SSLModel, MetricTracker
from self_supervised.datasets import *
import pytorch_lightning as pl
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def inference_pipeline(
        root_dir:str,
        outputs_dir:str,
        subject:str,
        classification_task:str='3-way',
        dataset_type_gen:str='generative_dataset',
        args:dict=None):
    
    
    imsize = args['imsize']
    batch_size = args['batch_size']
    seed = args['seed']
    
    results_dir = outputs_dir+subject+'/'+dataset_type_gen+'/'+classification_task+'/'
    model_dir = results_dir+'/best_model.ckpt'
    dataset_dir = root_dir+subject+'/'
    sslm = SSLM('3-way')
    sslm = SSLM.load_from_checkpoint(model_dir, model=sslm.model)
    print('')
    print('>>> Generating test dataset (artificial)')
    start = time.time()
    datamodule = GenerativeDatamodule(
                dataset_dir,
                imsize=imsize,
                batch_size=batch_size,
                seed=seed,
                classification_task='3-way',
                min_dataset_length=500,
                duplication=True
    )


    datamodule.setup('test')
    end = time.time() - start
    print('Generated in '+str(end)+ 'sec')
    
    print('>>> Inferencing...')
    x,y = next(iter(datamodule.test_dataloader())) 
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
    print(df)
    df.to_csv(results_dir+'/metric_report.csv', index = False)
    
    print('>>> Generating tsne visualization')
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(embeddings.detach().numpy())
    tx = tsne_results[:, 0]
    ty = tsne_results[:, 1]

    df = pd.DataFrame()
    df["labels"] = y
    df["comp-1"] = tx
    df["comp-2"] = ty
    plt.figure()
    sns.scatterplot(hue=df.labels.tolist(),
                    x='comp-1',
                    y='comp-2',
                    palette=sns.color_palette("hls", 3),
                    data=df).set(title='Embeddings projection ('+subject+', '+classification_task+')') 
    plt.savefig(results_dir+'/tsne.png')


if __name__ == "__main__":
    dataset_dir = 'dataset/'
    results_dir = 'outputs/computations/' 
    imsize=(256,256)
    batch_size = 64
    seed = 0
    
    args = {
        'imsize': imsize,
        'batch_size': batch_size,
        'seed': seed,
    }
    
    experiments = [
        ('bottle', '3-way', 'generative_dataset'),
        #('grid', '3-way', 'generative_dataset'),
        #('screw', '3-way', 'generative_dataset'),
        #('tile', '3-way', 'generative_dataset'),
        #('toothbrush', '3-way', 'generative_dataset')
    ]
    
    pbar = tqdm(range(len(experiments)))
    for i in pbar:
        pbar.set_description('Pipeline Execution | current subject is '+experiments[i][0].upper())
        inference_pipeline(
            dataset_dir, 
            results_dir, 
            experiments[i][0], 
            experiments[i][1], 
            experiments[i][2],
            args)
        os.system('clear')

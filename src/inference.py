from self_supervised.model import SSLM, SSLModel, MetricTracker
from self_supervised.datasets import *
import pytorch_lightning as pl
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt


subject = 'bottle'
dataset_type_gen = 'generative_dataset'
classification_task = '3-way'

results_dir = 'outputs/computations/'+subject+'/'+dataset_type_gen+'/'+classification_task+'/'
model_dir = results_dir+'/best_model.ckpt'
dataset_dir = 'dataset/'+subject
sslm = SSLM('3-way')
sslm = SSLM.load_from_checkpoint(model_dir, model=sslm.model)

random.seed(0)
np.random.seed(0)
print('generating dataset')
start = time.time()
datamodule = GenerativeDatamodule(
            dataset_dir,
            classification_task='3-way',
            min_dataset_length=500,
            duplication=True
)


datamodule.setup('test')
end = time.time() - start
print('generated in '+str(end)+ 'sec')

x,y = next(iter(datamodule.test_dataloader())) 
y_hat, embeddings = sslm(x)

y_hat = torch.max(y_hat.data, 1)
y_hat = y_hat.indices

result = classification_report( 
        y,
        y_hat,
        labels=[0,1,2],
        output_dict=True
    )
df = pd.DataFrame.from_dict(result)
print(df)
df.to_csv(results_dir+'/metric_report.csv', index = False)

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
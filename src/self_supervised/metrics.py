from sklearn.metrics import auc, roc_curve, f1_score
from torch import Tensor
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import os



def metrics_to_dataframe(metric_dict:dict, objects):
    df = pd.DataFrame(metric_dict, columns=metric_dict.keys(), index=objects)
    return df


def export_dataframe(dataframe:pd.DataFrame, saving_path:str=None, name:str='report.csv'):
    if saving_path and not os.path.exists(saving_path):
        os.makedirs(saving_path)
    if saving_path:
        dataframe.to_csv(saving_path+name)
    else:
        dataframe.to_csv(name)


def compute_f1(targets:Tensor, predictions:Tensor):
    f1 = f1_score(targets, predictions)
    return f1


def compute_optimal_f1(targets:Tensor, predictions:Tensor, thresholds:Tensor):
    max_f1 = 0
    for threshold in thresholds:
        x = predictions > threshold
        f1 = compute_f1(targets, x)
        if f1 > max_f1:
            max_f1 = f1
    return max_f1

def compute_roc(labels:Tensor, scores:Tensor):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    return false_positive_rate, true_positive_rate, thresholds


def compute_auc(false_positive_rate, true_positive_rate):
    roc_auc = auc(false_positive_rate, true_positive_rate)
    return roc_auc


def report(y:Tensor, y_hat:Tensor):
    result = classification_report( 
            y,
            y_hat,
            labels=[0,1,2],
            output_dict=True
        )
    df = pd.DataFrame.from_dict(result)
    return df.T



    
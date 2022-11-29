from sklearn.metrics import auc, roc_curve
from torch import Tensor
from sklearn.metrics import classification_report
import pandas as pd



def roc(labels:Tensor, scores:Tensor):
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    return false_positive_rate, true_positive_rate


def auc(false_positive_rate, true_positive_rate):
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
    return df



    
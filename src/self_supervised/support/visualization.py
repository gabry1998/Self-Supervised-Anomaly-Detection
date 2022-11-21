import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch import Tensor


def plot_history(network_history, epochs, saving_path='', mode='training'):
    x_plot = list(range(1,epochs+1))
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(x_plot, network_history['train']['loss'])
    plt.plot(x_plot, network_history['val']['loss'])
    plt.legend(['Training', 'Validation'])
    plt.savefig(saving_path+mode+'_loss.png')

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(x_plot, network_history['train']['accuracy'])
    plt.plot(x_plot, network_history['val']['accuracy'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.savefig(saving_path+mode+'_accuracy.png')
    plt.show()


def plot_roc(labels:Tensor, scores:Tensor, subject:str):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    #plot roc
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Roc curve ['+subject+']')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('test_roc.png')
    plt.close()


def localize_defect():
    pass
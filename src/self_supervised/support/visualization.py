import matplotlib.pyplot as plt

def plot_history(network_history, epochs, saving_path='', mode='binary'):
    x_plot = list(range(1,epochs+1))
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(x_plot, network_history['train']['loss'])
    plt.plot(x_plot, network_history['val']['loss'])
    plt.legend(['Training', 'Validation'])
    plt.savefig(saving_path+'loss.png')

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(x_plot, network_history['train']['accuracy'])
    plt.plot(x_plot, network_history['val']['accuracy'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.savefig(saving_path+'accuracy.png')
    plt.show()
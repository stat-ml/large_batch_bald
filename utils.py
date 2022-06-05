from torch import nn
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

def init_glorot(model):
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            
def plot_graph(PATH, algs, uns_type, dataset, acquisition_batch_size, random_seed, num_initial_samples, max_training_samples):
    plt.figure(figsize=(20, 15))
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 85
    
    alg_colors = {
    'PLBB': 'r',
    'PBALD': 'm',
    'Rand': 'c',
    'LBB': 'b', 
    'BALD': 'k',
    'BB': 'g'
    }
    
    for alg in algs:
        df = pd.read_csv('results/' + uns_type + "/" + dataset + "/" + str(acquisition_batch_size) + "/" + alg + str(random_seed) + ".csv")
        plt.plot(np.arange(num_initial_samples, max_training_samples + acquisition_batch_size - 1, acquisition_batch_size), df['Test accuracy'], alg_colors[alg], label=alg, linewidth=7.0)
    
    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")
    plt.title("Accuracy, " + dataset + ", batch " + str(acquisition_batch_size))
    plt.legend(fontsize=50)
    plt.savefig(PATH + '/' + uns_type.lower() + '_' + dataset.lower() + '_batch' + str(acquisition_batch_size) + '.svg')
#     plt.show()
    return
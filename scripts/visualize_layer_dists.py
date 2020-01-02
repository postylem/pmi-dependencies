# usage: 
# $ python visualize_layer_dists.py input_file.csv large
#
# see also on colab: visualize_dist_matrix.ipynb
# https://colab.research.google.com/drive/1zOAtKmkmwrJn4BzvBV-ONcmBJyGSwlIZ

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import io
from argparse import ArgumentParser

argp = ArgumentParser()
argp.add_argument('input_csv') # csv file to visualize
argp.add_argument('model_size', help='base or large')
args = argp.parse_args()

# Number of layers:
if args.model_size == 'large':
        NUM_LAYERS=24
elif args.model_size == 'base':
        NUM_LAYERS=12
else:
        raise ValueError("Error. Note model_size must be 'base' or 'large'.")

def visualize_dist_matrix(matrix, class_names, 
                           figsize = (9,6), fontsize=14, title='dist matrix'):
    """Plots a dist matrix"""
    df_cm = pd.DataFrame(
        matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="0.3f", cmap="coolwarm", vmin=0.5, vmax=0.9) # cmap="YlGnBu" for layer comparison or "coolwarm" for 2 layer probe
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='center')
    heatmap.set_ylim(len(class_names), 0)
    plt.ylabel('BERT-large layer')
    plt.xlabel('BERT-large layer')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# input csv file  
# or hardcode url such as 'https://raw.githubusercontent.com/postylem/comp551-miniproj4/master/tree_dists_BERT_large_dev.csv?token=ABHWIDKFR2M656AZT67D2C255G7ZU'

input_csv = args.input_csv
dists = pd.read_csv(input_csv,header=None).values
layers = [str(i+1) for i in range(NUM_LAYERS)]

visualize_dist_matrix(dists, layers, figsize = (9,6), fontsize=8, title='BERT-large deep probe on concatenated layers')


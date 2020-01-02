#_______________________________________________________________________________
#####READ THIS: This script will print and return an array where the [i,j] entry
# is the square similarity (i.e L^2 distance of difference of associated matrix)
# of two trees.

# Make sure you modify the directory at line 25, I wrote them as "BERT-n-4/dev.predictions")

#_____________________________________________________________

import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import glob
import os

argp = ArgumentParser()
argp.add_argument('input_path') # e.g. "results/BERT-disk*" for all directories beginning with BERT-disk
argp.add_argument('output_path') # e.g. tree_confusion_dist.csv
argp.add_argument('model_size', help='base or large')
args = argp.parse_args()

# Number of layers:
if args.model_size == 'large':
        NUM_LAYERS=24
elif args.model_size == 'base':
        NUM_LAYERS=12
else:
        raise ValueError("Error. Note model_size must be 'base' or 'large'.")

# Get the list of directories matching the input path substring (that is )
results_list = glob.glob(args.input_path)

data=[]

print(DIRECTORIES:)
for results_dir in results_list:
	print(results_dir)
	with open(results_dir+"/dev.predictions", "r") as f:
		data.append(np.array(json.load(f))) #stores the results

data=[d.flatten() for d in data] #forget batch stuff

#data should contain NUM_LAYERS lists of size equal to number of sentences in the file you are testing
#check:
print(len(data[0])==len(data[1])) 
print(len(data[1][1030])==len(data[0][1030]))
#print(len(data)==2)
#print(len(data[0])==1700)
#print(np.array(data[1][1030]).shape==(52, 52))

#each matrix/array in data[i] corresponds to the syntax tree obatined from layer i of a sentence 
result=np.zeros((len(data), len(data)))
for sent in tqdm(range(len(data[0])), desc='[computing distances]'):
	size_sent=len(data[0][sent]) #size of sentence
	result_sen=np.zeros((len(data), len(data)))
	for j in range(NUM_LAYERS):
		for i in range(NUM_LAYERS):
			#square difference for sentence sent of tree from layer i and tree from layer j
			square_diff=(np.array(data[i][sent])-np.array(data[j][sent]))*(np.array(data[i][sent])-np.array(data[j][sent]))
			#sum the squares, and normalize by size of sentence:
			result_sen[i, j]=np.sum(square_diff)/(square_diff.shape[0]*square_diff.shape[1])
	result+=result_sen

result=1/len(data[0])*result #normalize on number of sentence


print(result)
print('saving to',args.output_path)
np.savetxt(args.output_path, result, delimiter=',')


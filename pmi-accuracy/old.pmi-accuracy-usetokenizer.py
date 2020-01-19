"""
Cached version. Uses tokenizer...

Gets the undirected attachment accuracy score 
for dependencies calculated from PMI (using XLNet),
compared to gold dependency parses taken from a CONLL file.
-
Jacob Louis Hoover
January 2020
"""

# import os
import task
from tqdm import tqdm
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from collections import namedtuple#, defaultdict
from transformers import XLNetLMHeadModel, XLNetTokenizer
import torch
import torch.nn.functional as F

# Load up pretrained model and tokenizer.
MODEL = [(XLNetLMHeadModel, XLNetTokenizer, 'xlnet-base-cased')]
for model_class, tokenizer_class, pretrained_weights in MODEL:
  tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
  model = model_class.from_pretrained(pretrained_weights)

# Location of CONLL file with dependency labels (head_indices column)
FILEPATH = '/Users/j/McGill/PhD-miasma/xlnet-pmi/ptb3-wsj-data/ptb3-wsj-dev.conllx'

# Columns of CONLL file
FIELDNAMES = ['index',
              'sentence',
              'lemma_sentence',
              'upos_sentence',
              'xpos_sentence',
              'morph',
              'head_indices',
              'governance_relations',
              'secondary_relations',
              'extra_info']

ObservationClass = namedtuple("Observation", FIELDNAMES)

def generate_lines_for_sent(lines):
  '''Yields batches of lines describing a sentence in conllx.

  Args:
    lines: Each line of a conllx file.
  Yields:
    a list of lines describing a single sentence in conllx.
  '''
  buf = []
  for line in lines:
    if line.startswith('#'):
      continue
    if not line.strip():
      if buf:
        yield buf
        buf = []
      else:
        continue
    else:
      buf.append(line.strip())
  if buf:
    yield buf

def load_conll_dataset(filepath, observation_class):
  '''Reads in a conllx file; generates Observation objects

  For each sentence in a conllx file, generates a single Observation
  object.

  Args:
    filepath: the filesystem path to the conll dataset

  Returns:
  A list of Observations 
  '''
  observations = []
  lines = (x for x in open(filepath))
  for buf in generate_lines_for_sent(lines):
    conllx_lines = []
    for line in buf:
      conllx_lines.append(line.strip().split('\t'))
    # embeddings = [None for x in range(len(conllx_lines))]
    observation = observation_class(*zip(*conllx_lines)
      # ,embeddings
      )
    observations.append(observation)
  return observations


# The following functions are for getting PMI from XLNet

def tokenlist(sen):
  '''
  Input: sentence as string
  returns: simple list of tokens, according to tokenizer
  '''
  set_as_tokenlist = []
  input_ids = torch.tensor(tokenizer.encode(sen))
  for i in range(len(tokenizer.tokenize(sen))):
    set_as_tokenlist.append(tokenizer.decode(input_ids[i].item()))
  return set_as_tokenlist

BOLD, ULINE, PLAIN = '\033[1m', '\033[4m', '\033[0m' # just for formatting
def get_pmi(word1, word2, sentence, verbose=False):
  '''
  Input:
    word1,word2: word indices (integers)
    sentence: string
  returns: PMI(word1;word2|c), where c is the whole sentence except word1,word2
  '''
  input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
  # numerator = p(word1|word2,c) : predict word1 given the context (sentence with word1 masked)
  perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
  perm_mask[:, :, word1] = 1.0  # other tokens don't see target
  target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)
  target_mapping[0, 0, word1] = 1.0  # predict one token at location word1
  with torch.no_grad():
    logits_outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
    outputs = F.log_softmax(logits_outputs[0][0][0], 0)
    log_numerator = outputs[input_ids[0][word1].item()]
  # denominator = p(word1): predict word1 given sentence with word1 and word2 masked
  perm_mask2 = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
  perm_mask2[:, :, word1] = 1.0  # other tokens don't see the first target, word1
  perm_mask2[:, :, word2] = 1.0  # now other tokens don't see the second target word2 either.
  with torch.no_grad():
    logits_outputs = model(input_ids, perm_mask=perm_mask2, target_mapping=target_mapping)
    outputs = F.log_softmax(logits_outputs[0][0][0], 0)
    log_denominator = outputs[input_ids[0][word1].item()]
  pmi = (log_numerator - log_denominator).item()
  if verbose:
    tokens = tokenlist(sentence)
    if word1 < word2:
      first, second, style1, style2 = word1, word2, BOLD, ULINE
    else: first, second, style1, style2 = word2, word1, ULINE, BOLD
    print("sentence: ", ' '.join(tokens[:first]),
          ' ', style1, tokens[first], PLAIN, ' ',
          ' '.join(tokens[first+1:second]),
          ' ', style2, tokens[second], PLAIN, ' ',
          ' '.join(tokens[second+1:]),
          sep='')
    print("log p(", BOLD, tokenizer.decode(input_ids[0][word1].item()),
          PLAIN, "|", ULINE, tokenizer.decode(input_ids[0][word2].item()),
          PLAIN, ", rest of sentence) = %.4f"%log_numerator.item(), sep='')
    print("log p(", BOLD, tokenizer.decode(input_ids[0][word1].item()),
          PLAIN, "|rest of sentence) = %.4f"%log_denominator.item(),sep='')
    print("PMI = %.4f"%pmi)
  return pmi

def get_pmi_matrix(sen, verbose=False):
  ''' Input: sentence (string). Returns: an ndArray of PMIs '''
  slength = len(tokenizer.tokenize(sen))
  input_ids = torch.tensor(tokenizer.encode(sen))
  # print out the tokenized sentence
  if verbose:
    print("Getting PMI matrix for sentence:")
    for i in range(slength):
      print('%i:%s'%(i, tokenizer.decode(input_ids[i].item())), end='|')
    print()
  # pmi matrix
  pmis = np.ndarray(shape=(slength, slength))
  for word1_id in tqdm(range(slength), desc='[get_pmi_matrix]'):
    # compute row of pmis given word1
    for word2_id in tqdm(range(slength),
                         leave=False,
                         desc=f'{word1_id}:{tokenizer.decode(input_ids[word1_id].item())}'):
      if word2_id == word1_id:
        pmis[word1_id][word2_id] = np.NaN
      else:
        pmis[word1_id][word2_id] = get_pmi(word1_id, word2_id, sen, verbose=False)
  return pmis

class UnionFind:
  '''
  Naive UnionFind (for computing MST with Prim's algorithm).
  '''
  def __init__(self, n):
    self.parents = list(range(n))
  def union(self, i, j):
    '''
    join i and j
    '''
    if self.find(i) != self.find(j):
      i_parent = self.find(i)
      self.parents[i_parent] = j
  def find(self, i):
    '''
    find parent of i
    '''
    i_parent = i
    while True:
      if i_parent != self.parents[i_parent]:
        i_parent = self.parents[i_parent]
      else:
        break
    return i_parent

def prims_matrix_to_edges(matrix, words, maximum_spanning_tree=True, verbose=False):
  '''
  Constructs a maximum spanning tree using Prim's algorithm.
  Input: matrix (ndArray of PMIs), tokenlist words corresponding list of tokens
  (set maximum_spanning_tree=False to get minumum spanning tree instead).
  Excludes edges to/from punctuation symbols or empty strings.
  Returns: tree (list of edges).
  By John Hewitt, modified.
  '''
  pairs_to_weights = {}
  union_find = UnionFind(len(matrix))
  for i_index, line in enumerate(matrix):
    for j_index, dist in enumerate(line):
      if words[i_index] in ["", "'", "''", ",", ".", ":", "``", "-LRB-", "-RRB-"]:
        continue
      if words[j_index] in ["", "'", "''", ",", ".", ":", "``", "-LRB-", "-RRB-"]:
        continue
      pairs_to_weights[(i_index, j_index)] = dist
  edges = []
  for (i_index, j_index), weight in sorted(pairs_to_weights.items(), key=lambda x: x[1],
                                           reverse=maximum_spanning_tree):
    if union_find.find(i_index) != union_find.find(j_index):
      union_find.union(i_index, j_index)
      edges.append((i_index, j_index))
      if verbose:
        print(f'({i_index}, {j_index}) : {weight}')
  return edges

def get_edges_from_matrix(matrix, words, symmetrize_method='sum', verbose=False):
  '''
  Gets Maximum Spanning Tree (list of edges) from pmi matrix, using the specified method.
  input:
    matrix: an array of pmis
    words: a list of tokens
    symmetrize_method:
      'sum' (default): sums matrix with transpose of matrix;
      'triu': uses only the upper triangle of matrix;
      'tril': uses only the lower triangle of matrix;
      'none': uses the optimum weight for each unordered pair of edges.
  returns: tree (list of edges)
  '''

  if symmetrize_method == 'sum':
    matrix = matrix + np.transpose(matrix)
  elif symmetrize_method == 'triu':
    matrix = np.triu(matrix) + np.transpose(np.triu(matrix))
  elif symmetrize_method == 'tril':
    matrix = np.tril(matrix) + np.transpose(np.tril(matrix))
  elif symmetrize_method != 'none':
    raise ValueError("Unknown symmetrize_method. Use 'sum', 'triu', 'tril', or 'none'")

  if verbose:
    print(f'Getting MST from matrix, using symmetrize_method = {symmetrize_method}.')
  edges = prims_matrix_to_edges(matrix, words, maximum_spanning_tree=True, verbose=verbose)
  return edges

def get_uuas(observation, verbose=False):
  '''
  gets the unlabeled undirected attachment score for a given sentence (observation),
  by reading off the minimum spanning tree from a matrix of PTB dependency distances
  and comparing that to the maximum spanning tree from a matrix of PMIs
  '''
  if verbose:
    obs_df = pd.DataFrame(observation).T
    obs_df.columns = FIELDNAMES
    print("Gold observation\n", obs_df.loc[:, ['index', 'sentence', 'head_indices']], sep='')

  # Get gold edges from conllx file
  gold_dist_matrix = task.ParseDistanceTask.labels(observation)
  gold_edges = prims_matrix_to_edges(gold_dist_matrix, observation.sentence,
                                     maximum_spanning_tree=False)

  # Calculate pmi edges from XLNet
  xlnet_sentence = ' '.join(observation.sentence)
  print(f"Joined xlnet_sentence, on which XLNetTokenizer will be run:\n{xlnet_sentence}")
  pmi_matrix = get_pmi_matrix(xlnet_sentence, verbose=True)
  pmi_edges = get_edges_from_matrix(pmi_matrix, tokenlist(xlnet_sentence),
                                    symmetrize_method='sum', verbose=True)

  print("gold:", gold_edges)
  print("pmi :", pmi_edges)
  correct = {tuple(sorted(x)) for x in gold_edges}.intersection(
    {tuple(sorted(x)) for x in pmi_edges})
  print(correct)
  num_correct = len(correct)
  num_total = len(gold_edges)
  uuas = num_correct/float(num_total)
  print(f'uuas = #correct / #total = {num_correct}/{num_total} = {uuas}')
  return uuas


if __name__ == '__main__':

  OBSERVATIONS = load_conll_dataset(FILEPATH, ObservationClass)
  UUAS = get_uuas(OBSERVATIONS[20])

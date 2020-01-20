"""
Gets the undirected attachment accuracy score
for dependencies calculated from PMI (using XLNet),
compared to gold dependency parses taken from a CONLL file.

Default: run from directory where conllx file
with dependency labels (head_indices column) is located at
ptb3-wsj-data/ptb3-wsj-dev.conllx
-
Jacob Louis Hoover
January 2020
"""

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import csv
# import matplotlib.pyplot as plt
from datetime import datetime
from argparse import ArgumentParser
from collections import namedtuple#, defaultdict

import torch
import torch.nn.functional as F

import task

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

def get_pmi_matrix_from_ids(sentence_as_ids, verbose=False):
  '''
  Input:
    sentence_as_ids: a torch.tensor of ids ending with 4,3 for <sep><cls>
      (like the output of tokenizer.encode())
  returns:
    ndarray of PMIs PMI(w1;w2|c) for all values of w1,w2 from 0 up to sentence length:
      (where c is the whole sentence except w1,w2)
  '''
  if verbose:
    # print out the tokenized sentence
    print("Getting PMI matrix for sentence:")
    for i, input_id in enumerate(sentence_as_ids):
      print(f'{i}:{tokenizer.decode(input_id.item())}|', end='')
    print()

  seqlen = sentence_as_ids.shape[0]
  input_ids = sentence_as_ids.unsqueeze(0).repeat(seqlen**2, 1)

  perm_mask = torch.zeros((seqlen**2, seqlen, seqlen), dtype=torch.float)
  target_mapping = torch.zeros((seqlen**2, 1, seqlen), dtype=torch.float)
  for w1 in range(seqlen):
    perm_mask[(w1*seqlen):((w1+1)*seqlen), :, w1] = 1.0  # other tokens don't see w1 (target)
    target_mapping[(w1*seqlen):((w1+1)*seqlen), :, w1] = 1.0 # predict just w1
    for w2 in range(seqlen):
      perm_mask[(w1*seqlen)+w2, :, w2] = 1 # for denominator only: other tokens don't see w2

  with torch.no_grad():
    # gives tuple ([seqlen**2,1,vocabsize],)
    logits_outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
    # log softmax across the vocabulary (dimension 2 of the logits tensor)
    outputs = F.log_softmax(logits_outputs[0], 2)
    # reshape to be tensor.Size[seqlen,seqlen,vocabsize]
    outputs = outputs.view(seqlen, seqlen, -1)

  #only up to seqlen-2, because sentence_as_ids contains the extra <sep><cls> at the end
  pmis = np.ndarray(shape=(seqlen-2, seqlen-2))
  for w1 in range(seqlen-2):
    for w2 in range(seqlen-2):
      log_numerator2   = outputs[w1][w1][input_ids[1][w1].item()]
      log_denominator2 = outputs[w1][w2][input_ids[1][w1].item()]
      pmis[w1][w2] = (log_numerator2 - log_denominator2).item()
  return pmis

def make_sentencepiece_tokenlist(ptb_tokenlist):
  '''
  Takes list of tokens from plaintext of Treebank, formats them
  as if they were the sentencepiece tokens expected by XLNet
  '''
  sentencepiece_tokenlist = []
  for token in ptb_tokenlist:
    if token[0].isalpha():
      token = '▁' + token
    elif token[0] == "'":
      token = '('
    elif token == '-LRB-':
      token = '('
    elif token == '-RRB-':
      token = ')'
    sentencepiece_tokenlist.append(token)
  sentencepiece_tokenlist.append('<sep><cls>')
  return sentencepiece_tokenlist

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

def prims_matrix_to_edges(matrix, words, maximum_spanning_tree=True):
  '''
  Constructs a maximum spanning tree using Prim's algorithm.
    (set maximum_spanning_tree=False to get minumum spanning tree instead).
  Input: matrix (ndArray of PMIs), words (list of tokens)
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
  for (i_index, j_index), _ in sorted(pairs_to_weights.items(), key=lambda x: x[1],
                                      reverse=maximum_spanning_tree):
    if union_find.find(i_index) != union_find.find(j_index):
      union_find.union(i_index, j_index)
      edges.append((i_index, j_index))
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
  edges = prims_matrix_to_edges(matrix, words, maximum_spanning_tree=True)
  return edges

def get_uuas_for_observation(observation, use_tokenizer=False, verbose=False):
  '''
  gets the unlabeled undirected attachment score for a given sentence (observation),
  by reading off the minimum spanning tree from a matrix of PTB dependency distances
  and comparing that to the maximum spanning tree from a matrix of PMIs
  (TODO, this is not worth much now:)
  set use_tokenizer=True to use XLNet tokenizer
  '''
  if verbose:
    obs_df = pd.DataFrame(observation).T
    obs_df.columns = FIELDNAMES
    print("Gold observation\n", obs_df.loc[:, ['index', 'sentence', 'head_indices']], sep='')

  # Get gold edges from conllx file
  gold_dist_matrix = task.ParseDistanceTask.labels(observation)
  gold_edges = prims_matrix_to_edges(gold_dist_matrix, observation.sentence,
                                     maximum_spanning_tree=False)
  # tokenize sentence
  if use_tokenizer:
    xlnet_sentence = ' '.join(observation.sentence)
    print(f"Joined xlnet_sentence, on which XLNetTokenizer will be run:\n{xlnet_sentence}")
    tokenlist = tokenizer.tokenize(xlnet_sentence)
  else: tokenlist = make_sentencepiece_tokenlist(observation.sentence)

  # Calculate pmi edges from XLNet
  input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenlist))
  pmi_matrix = get_pmi_matrix_from_ids(input_ids, verbose=verbose)
  pmi_edges = {}
  symmetrize_methods = ['sum', 'triu', 'tril', 'none']
  for symmetrize_method in symmetrize_methods:
    pmi_edges[symmetrize_method] = get_edges_from_matrix(
      pmi_matrix, tokenlist, symmetrize_method=symmetrize_method, verbose=verbose)

  scores = []
  gold_edges_set = {tuple(sorted(x)) for x in gold_edges}
  print("gold set:", gold_edges_set)
  for symmetrize_method in symmetrize_methods:
    pmi_edges_set = {tuple(sorted(x)) for x in pmi_edges[symmetrize_method]}
    print(f'pmi {symmetrize_method:4}: {pmi_edges_set}')
    correct = gold_edges_set.intersection(pmi_edges_set)
    print("correct: ", correct)
    num_correct = len(correct)
    num_total = len(gold_edges)
    uuas = num_correct/float(num_total)
    scores.append(uuas)
    print(f'uuas = #correct / #total = {num_correct}/{num_total} = {uuas:.3f}')
  return scores

def report_uuas_n(observations, n_observations, results_dir, verbose=False):
  '''
  Draft version.
  Gets the uuas for observations[0:n_observations]
  Writes to scores and mean_scores csv files.
  Outputs array mean_scores for [sum, triu, tril, none]
  '''
  results_filepath = os.path.join(results_dir, 'scores.csv')
  all_scores = []
  with open(results_filepath, mode='w') as results_file:
    results_writer = csv.writer(results_file, delimiter=',')
    for i, observation in enumerate(tqdm(observations[:n_observations])):
      scores = get_uuas_for_observation(observation, use_tokenizer=False, verbose=verbose)
      results_writer.writerow([i+1, len(observation.sencence), scores[0], scores[1], scores[2], scores[3]])
      all_scores.append(scores)
  mean_scores = [float(sum(col))/len(col) for col in zip(*all_scores)]
  if verbose:
    print(f'\n---\nmean_scores[sum,triu,tril,none]: {mean_scores}')
  return mean_scores



if __name__ == '__main__':
  ARGP = ArgumentParser()
  ARGP.add_argument('--n_observations', type=int, default='20',
                    help='number of sentences to look at')
  ARGP.add_argument('--offline-mode', action='store_true',
                    help='set for "pytorch-transformers" (specify path in xlnet-spec)')
  ARGP.add_argument('--xlnet-spec', default='xlnet-base-cased',
                    help='specify "xlnet-base-cased" or "xlnet-large-cased", or path')
  ARGP.add_argument('--conllx-file', default='ptb3-wsj-data/ptb3-wsj-test.conllx',
                    help='path/to/treebank.conllx: dependency file, in conllx format')
  ARGP.add_argument('--results-dir', default='results/',
                    help='specify path/to/results/directory/')
  CLI_ARGS = ARGP.parse_args()

  if CLI_ARGS.offline_mode:
    # import pytorch transformers instead of transformers, for use on Compute Canada
    from pytorch_transformers import XLNetLMHeadModel, XLNetTokenizer
    SPEC_STRING = 'offline'
  else:
    from transformers import XLNetLMHeadModel, XLNetTokenizer
    SPEC_STRING = str(CLI_ARGS.xlnet_spec)

  NOW = datetime.now()
  DATE_SUFFIX = f'{NOW.year}-{NOW.month:02}-{NOW.day:02}-{NOW.hour:02}-{NOW.minute:02}'
  SPEC_SUFFIX = SPEC_STRING+str(CLI_ARGS.n_observations)
  RESULTS_DIR = os.path.join(CLI_ARGS.results_dir, SPEC_SUFFIX + '_' + DATE_SUFFIX + '/')
  os.makedirs(RESULTS_DIR, exist_ok=True)
  print(f'RESULTS_DIR: {RESULTS_DIR}\n')

  print('Running pmi-accuracy.py with cli arguments:')
  with open(RESULTS_DIR+'spec.txt', mode='w') as specfile:
    for arg, value in sorted(vars(CLI_ARGS).items()):
      specfile.write(f"\t{arg}:\t{value}\n")
      print(f"\t{arg}:\t{value}")
    specfile.write("scores.csv header row: [sentence_index, sentence_length, sym=sum, sym=triu, sym=tril, sym=none]")
    specfile.close()

  MODEL = [(XLNetLMHeadModel, XLNetTokenizer, CLI_ARGS.xlnet_spec)]
  for model_class, tokenizer_class, pretrained_weights in MODEL:
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

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

  OBSERVATIONS = load_conll_dataset(CLI_ARGS.conllx_file, ObservationClass)

  MEAN_SCORES = report_uuas_n(OBSERVATIONS, CLI_ARGS.n_observations, RESULTS_DIR, verbose=True)
  with open(RESULTS_DIR+'mean_scores.csv', mode='w') as file:
    csv.writer(file, delimiter=',').writerow(MEAN_SCORES)

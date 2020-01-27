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

def get_pmi_matrix_from_ids(sentence_as_ids, device, verbose=False):
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
    # model() gives tuple ([seqlen**2,1,vocabsize],)
    logits_outputs = model(input_ids.to(device),
                           perm_mask=perm_mask.to(device),
                           target_mapping=target_mapping.to(device))
    # log softmax across the vocabulary (dimension 2 of the logits tensor)
    outputs = F.log_softmax(logits_outputs[0], 2)
    # reshape to be tensor.Size[seqlen,seqlen,vocabsize]
    outputs = outputs.view(seqlen, seqlen, -1)

  #only up to seqlen-2, because sentence_as_ids contains the extra <sep><cls> at the end
  input_ids = input_ids.cpu().numpy() # .cpu() shouldn't strictly be necessary
  outputs = outputs.cpu().numpy()
  pmis = np.ndarray(shape=(seqlen-2, seqlen-2))
  for w1 in range(seqlen-2):
    for w2 in range(seqlen-2):
      log_numerator2   = outputs[w1][w1][input_ids[1][w1]]
      log_denominator2 = outputs[w1][w2][input_ids[1][w1]]
      pmis[w1][w2] = (log_numerator2 - log_denominator2)
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
      token = token # "'s" should really be "'","s"
    elif token == '-LRB-':
      token = '('
    elif token == '-RRB-':
      token = ')'
    sentencepiece_tokenlist.append(token)
  sentencepiece_tokenlist.append('<sep>')
  sentencepiece_tokenlist.append('<cls>')
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

def score_observation(observation, device, use_tokenizer=False, verbose=False):
  '''
  gets the unlabeled undirected attachment score for a given sentence (observation),
  by reading off the minimum spanning tree from a matrix of PTB dependency distances
  and comparing that to the maximum spanning tree from a matrix of PMIs
  specify 'cuda' or 'cpu' as device
  (TODO, this is not worth much now:)
  set use_tokenizer=True to use XLNet tokenizer
  returns: number_of_unks (int), list_of_scores (list of floats)
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
  pmi_matrix = get_pmi_matrix_from_ids(input_ids, device=device, verbose=verbose)
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
    uuas = num_correct/float(num_total) if num_total != 0 else np.NaN
    scores.append(uuas)
    print(f'uuas = #correct / #total = {num_correct}/{num_total} = {uuas:.3f}')
  number_of_unks = tokenizer.convert_tokens_to_ids(tokenlist).count(0)
  return scores, number_of_unks

def report_uuas_n(observations, results_dir, device, n_obs='all', verbose=False):
  '''
  Draft version.
  Gets the uuas for observations[0:n_obs]
  Writes to scores and mean_scores csv files.
  Returns: list of mean_scores for [sum, triu, tril, none] (ignores NaN values)
  '''
  results_filepath = os.path.join(results_dir, 'scores.csv')
  all_scores = []
  with open(results_filepath, mode='w') as results_file:
    results_writer = csv.writer(results_file, delimiter=',')
    results_writer.writerow(['sentence_index', 'sentence_length', 'unks',
                             'uuas_sum', 'uuas_triu', 'uuas_tril', 'uuas_none'])
    if n_obs == 'all':
      n_obs = len(observations)

    long_sentence_count = 0
    for i, observation in enumerate(tqdm(observations[:n_obs])):
      if len(observation.sentence) > 55:
        # Hack for now, to skip long sentences
        print(f"SKIPPING LONG SENTENCE ({len(observation.sentence)} tokens)")
        scores = [np.NaN,np.NaN,np.NaN,np.NaN]
        unks = np.NaN
        long_sentence_count += 1
      else: 
        scores, unks = score_observation(observation, device=device, use_tokenizer=False, verbose=verbose)
      results_writer.writerow([i, len(observation.sentence), unks,
                               scores[0], scores[1], scores[2], scores[3]])
      all_scores.append(scores)
  # return means as list
  mean_scores = np.nanmean(np.array(all_scores), axis=0).tolist()
  if verbose:
    print(f'\n---\nmean_scores[sum,triu,tril,none]: {mean_scores}')
  return n_obs, long_sentence_count, mean_scores


if __name__ == '__main__':
  ARGP = ArgumentParser()
  ARGP.add_argument('--n_observations', default='all',
                    help='number of sentences to look at')
  ARGP.add_argument('--offline-mode', action='store_true',
                    help='set for "pytorch-transformers" (specify path in xlnet-spec)')
  ARGP.add_argument('--xlnet-spec', default='xlnet-base-cased',
                    help='specify "xlnet-base-cased" or "xlnet-large-cased", or path')
  ARGP.add_argument('--conllx-file', default='ptb3-wsj-data/ptb3-wsj-dev.conllx',
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

  N_OBS = CLI_ARGS.n_observations
  if N_OBS != 'all':
    N_OBS = int(N_OBS)

  NOW = datetime.now()
  DATE_SUFFIX = f'{NOW.year}-{NOW.month:02}-{NOW.day:02}-{NOW.hour:02}-{NOW.minute:02}'
  SPEC_SUFFIX = SPEC_STRING+str(CLI_ARGS.n_observations) if CLI_ARGS.n_observations != 'all' else SPEC_STRING
  RESULTS_DIR = os.path.join(CLI_ARGS.results_dir, SPEC_SUFFIX + '_' + DATE_SUFFIX + '/')
  os.makedirs(RESULTS_DIR, exist_ok=True)
  print(f'RESULTS_DIR: {RESULTS_DIR}\n')

  print('Running pmi-accuracy.py with cli arguments:')
  with open(RESULTS_DIR+'spec.txt', mode='w') as specfile:
    for arg, value in sorted(vars(CLI_ARGS).items()):
      specfile.write(f"\t{arg}:\t{value}\n")
      print(f"\t{arg}:\t{value}")
    specfile.close()

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

  DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('Using device:', DEVICE)
  if DEVICE.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')

  MODEL = [(XLNetLMHeadModel, XLNetTokenizer, CLI_ARGS.xlnet_spec)]
  for model_class, tokenizer_class, pretrained_weights in MODEL:
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    model = model.to(DEVICE)

  N_SENTS, SKIPPED, MEANS = report_uuas_n(OBSERVATIONS, RESULTS_DIR, n_obs=N_OBS, device=DEVICE, verbose=True)
  with open(RESULTS_DIR+'mean_scores.csv', mode='w') as file:
    WRITER = csv.writer(file, delimiter=',')
    WRITER.writerow(['n_sentences', 'total_skipped',
                     'mean_sum', 'mean_triu', 'mean_tril', 'mean_none'])
    WRITER.writerow([N_SENTS, SKIPPED,
                     MEANS[0], MEANS[1], MEANS[2], MEANS[3]])

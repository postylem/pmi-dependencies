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


def string_to_tokenlist(sen):
  '''
  Input: sentence as string
  returns: simple list of tokens, according to tokenizer
  '''
  sentence_tokenlist = []
  input_ids = torch.tensor(tokenizer.encode(sen))
  for i in range(len(tokenizer.tokenize(sen))):
    sentence_tokenlist.append(tokenizer.decode(input_ids[i].item()))
  return sentence_tokenlist

def get_logprob_masked_word(input_ids, index1):
  '''
  Computes log p( index1 | sentence with index1 masked )
  '''
  perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
  perm_mask[:, :, index1] = 1.0  # other tokens don't see target
  target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)
  target_mapping[0, 0, index1] = 1.0  # predict one token at location index1
  with torch.no_grad():
    logits_outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
    outputs = F.log_softmax(logits_outputs[0][0][0], 0)
    logprob = outputs[input_ids[0][index1].item()]
  return logprob

def get_logprob_masked_word_with_second_masked_word(input_ids, index1, index2):
  '''
  Computes log p( index1 | sentence with index1 and index2 masked )
  '''
  perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
  perm_mask[:, :, index1] = 1.0  # other tokens don't see the first target, index1
  perm_mask[:, :, index2] = 1.0  # now other tokens don't see the second target word either.
  target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)
  target_mapping[0, 0, index1] = 1.0  # predict one token at location index1
  with torch.no_grad():
    logits_outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
    outputs = F.log_softmax(logits_outputs[0][0][0], 0)
    logprob = outputs[input_ids[0][index1].item()]
  return logprob

def get_pmi_from_idlist(word1, word2, input_ids, verbose=False):
  '''
  Input:
    word1,word2: word indices (integers)
    input_ids: list of ids (for tokenizer to decode)
  returns: PMI(word1;word2|c), where c is the whole sentence except word1,word2
  '''
  input_ids = input_ids.unsqueeze(0) # because XLNet expects this format
  # numerator = p(word1|word2,c) : predict word1 given the context (sentence with word1 masked)
  log_numerator = get_logprob_masked_word(input_ids, word1)
  # denominator = p(word1): predict word1 given sentence with word1 and word2 masked
  log_denominator = get_logprob_masked_word_with_second_masked_word(input_ids, word1, word2)
  pmi = (log_numerator - log_denominator).item()
  if verbose:
    print("log p(", '\033[1m', tokenizer.decode(input_ids[0][word1].item()),
          '\033[0m', "|", '\033[4m', tokenizer.decode(input_ids[0][word2].item()),
          '\033[0m', ", rest of sentence) = %.4f"%log_numerator.item(), sep='')
    print("log p(", '\033[1m', tokenizer.decode(input_ids[0][word1].item()),
          '\033[0m', "|rest of sentence) = %.4f"%log_denominator.item(),sep='')
    print("PMI = %.4f"%pmi)
  return pmi

def get_pmi_matrix_from_idlist(input_ids, verbose=False):
  '''
  Input: input_ids (list).
  Returns: an ndArray of PMIs
  '''
  # print out the tokenized sentence
  if verbose:
    print("Getting PMI matrix for sentence:")
    for i, input_id in enumerate(input_ids):
      print(f'{i}:{tokenizer.decode(input_id.item())}|', end='')
    print()
  # pmi matrix
  pmis = np.ndarray(shape=(len(input_ids), len(input_ids)))
  for word1_id in tqdm(range(len(input_ids)), desc='[get_pmi_matrix]'):
    # compute row of pmis given word1
    for word2_id in tqdm(range(len(input_ids)),
                         leave=False,
                         desc=f'{word1_id}:{tokenizer.decode(input_ids[word1_id].item())}'):
      if word2_id == word1_id:
        pmis[word1_id][word2_id] = np.NaN
      else:
        pmis[word1_id][word2_id] = get_pmi_from_idlist(word1_id, word2_id, input_ids, verbose=False)
  return pmis

def get_pmi_from_sentence(word1, word2, sentence, verbose=False):
  '''
  [ALERT! uses XLNet tokenizer to get tokens..]
  Input:
    word1,word2: word indices (integers)
    sentence: string
  returns: PMI(word1;word2|c), where c is the whole sentence except word1,word2
  '''
  input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
  pmi = get_pmi_from_idlist(word1, word2, input_ids, verbose=verbose)

  if verbose:
    tokens = string_to_tokenlist(sentence)
    if word1 < word2:
      first, second, style1, style2 = word1, word2, '\033[1m', '\033[4m'
    else: first, second, style1, style2 = word2, word1, '\033[4m', '\033[1m'
    print("sentence: ", ' '.join(tokens[:first]),
          ' ', style1, tokens[first], '\033[0m', ' ',
          ' '.join(tokens[first+1:second]),
          ' ', style2, tokens[second], '\033[0m', ' ',
          ' '.join(tokens[second+1:]),
          sep='')
  return pmi

def get_pmi_matrix_from_sentence(sen, verbose=False):
  '''
  [ALERT! Uses XLNet tokenizer to get tokens]
  Input: sentence (string).
  Returns: an ndArray of PMIs
  '''
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
        pmis[word1_id][word2_id] = get_pmi_from_sentence(word1_id, word2_id, sen, verbose=False)
  return pmis

def make_sentencepiece_tokenlist(ptb_tokenlist):
  '''
  Takes list of tokens from the Treebank, formats them
  as if they were sentencepiece tokens expected by XLNet
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
  set use_tokenizer=True to use XLNet tokenizer (TODO, this is not worth much now)
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
  if use_tokenizer:
    xlnet_sentence = ' '.join(observation.sentence)
    print(f"Joined xlnet_sentence, on which XLNetTokenizer will be run:\n{xlnet_sentence}")
    tokenlist = tokenizer.tokenize(xlnet_sentence)
  else: tokenlist = make_sentencepiece_tokenlist(observation.sentence)

  input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenlist))
  pmi_matrix = get_pmi_matrix_from_idlist(input_ids, verbose=verbose)
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

def report_uuas_batch(observations, batch_size, results_dir, verbose=False):
  '''
  Draft version.
  Gets the uuas for observations[0:batch_size]
  Writes to scores and mean_scores csv files.
  Outputs array mean_scores for [sum, triu, tril, none]
  '''
  results_filepath = results_dir+'scores.csv'
  all_scores = []
  with open(results_filepath, mode='w') as results_file:
    results_writer = csv.writer(results_file, delimiter=',')
    for i, observation in enumerate(tqdm(observations)):
      if i+1 > batch_size:
        break
      scores = get_uuas_for_observation(observation, use_tokenizer=False, verbose=verbose)
      results_writer.writerow([i+1, scores[0], scores[1], scores[2], scores[3]])
      all_scores.append(scores)
  mean_scores = [float(sum(col))/len(col) for col in zip(*all_scores)]
  if verbose:
    print(f'\n---\nmean_scores[sum,triu,tril,none]: {mean_scores}')
  return mean_scores



if __name__ == '__main__':
  ARGP = ArgumentParser()
  ARGP.add_argument('--batch-size', type=int, default='20')
  ARGP.add_argument('--offline-mode', action='store_true',
                    help='set for "pytorch-transformers" (specify path in xlnet-spec)')
  ARGP.add_argument('--xlnet-spec', default='xlnet-base-cased',
                    help='specify "xlnet-base-cased" or "xlnet-large-cased", or path')
  ARGP.add_argument('--conllx-file', default='ptb3-wsj-data/ptb3-wsj-test.conllx',
                    help='path to PTB dependency file in conllx format')
  ARGP.add_argument('--results-dir', default='results/',
                    help='path to results directory, ending with /')
  CLI_ARGS = ARGP.parse_args()

  if CLI_ARGS.offline_mode:
    from pytorch_transformers import XLNetLMHeadModel, XLNetTokenizer
    SPEC_STRING = 'offline'
  else:
    from transformers import XLNetLMHeadModel, XLNetTokenizer
    SPEC_STRING = str(CLI_ARGS.xlnet_spec)

  NOW = datetime.now()
  DATE_SUFFIX = f'{NOW.year}-{NOW.month:02}-{NOW.day:02}-{NOW.hour:02}-{NOW.minute:02}'
  SPEC_SUFFIX = SPEC_STRING+str(CLI_ARGS.batch_size)
  RESULTS_DIR = os.path.join(CLI_ARGS.results_dir, SPEC_SUFFIX + '_' + DATE_SUFFIX + '/')
  os.makedirs(RESULTS_DIR, exist_ok=True)
  print(f'RESULTS_DIR: {RESULTS_DIR}\n')

  print('Running pmi-accuracy.py with cli arguments:')
  with open(RESULTS_DIR+'cli-args.txt', mode='w') as specfile:
    for arg, value in sorted(vars(CLI_ARGS).items()):
      specfile.write(f"\t{arg}:\t{value}\n")
      print(f"\t{arg}:\t{value}")
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

  MEAN_SCORES = report_uuas_batch(OBSERVATIONS, CLI_ARGS.batch_size, RESULTS_DIR, verbose=True)
  with open(RESULTS_DIR+'mean_scores.csv', mode='w') as file:
    csv.writer(file, delimiter=',').writerow(MEAN_SCORES)

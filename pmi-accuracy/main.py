import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import csv
import shutil
from datetime import datetime
from argparse import ArgumentParser
from collections import namedtuple

import torch

import task
import parser
import languagemodel

# Data input
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
    observation_class: namedtuple for observations

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

# Running and reporting
def score_observation(observation, pmi_matrix):
  # Get gold edges distances tensor from conllx file (note 'mst' will always give projective gold edges)
  gold_dist_matrix = task.ParseDistanceTask.labels(observation)
  gold_edges = parser.DepParse(
      'mst', gold_dist_matrix, observation.sentence).tree(
          symmetrize_method='none',
          maximum_spanning_tree=False)

  # Make linear-order baseline distances tensor
  linear_dist_matrix = task.LinearBaselineTask.labels(observation)
  baseline_linear_edges = parser.DepParse(
      'mst', linear_dist_matrix, observation.sentence).tree(
          symmetrize_method='none',
          maximum_spanning_tree=False)
  
  # Make random baseline distances tensor
  random_dist_matrix = task.RandomBaselineTask.labels(observation)
  baseline_random_nonproj_edges = parser.DepParse(
      'mst', random_dist_matrix, observation.sentence).tree(
          symmetrize_method='none',
          maximum_spanning_tree=False)
  baseline_random_proj_edges = parser.DepParse(
      'projective', random_dist_matrix, observation.sentence).tree(
          symmetrize_method='none',
          maximum_spanning_tree=True)

  # Instantiate a parser.DepParse object, with the parsetype 'mst', to get pmi mst parse
  mstparser = parser.DepParse('mst', pmi_matrix, observation.sentence)
  pmi_edges = {}
  symmetrize_methods = ['sum', 'triu', 'tril', 'none']
  for symmetrize_method in symmetrize_methods:
    pmi_edges[symmetrize_method] = mstparser.tree(symmetrize_method=symmetrize_method)

  # Instantiate a parser.DepParse object, with parsetype 'projective', to get pmi projective parse
  projparser = parser.DepParse('projective', pmi_matrix, observation.sentence)
  pmi_edges_proj = {}
  for symmetrize_method in symmetrize_methods:
    # note, with Eisner's, symmetrize_method='none' basically gets a directed parse
    pmi_edges_proj[symmetrize_method] = projparser.tree(symmetrize_method=symmetrize_method)

  scorer = parser.Accuracy(gold_edges)

  scores = {}
  scores['sentence_length'] = len(observation.sentence)
  scores['number_edges'] = len(gold_edges)
  scores['gold_edges'] = gold_edges
  scores['baseline_linear'] = scorer.uuas(baseline_linear_edges)
  scores['baseline_random_nonproj'] = scorer.uuas(baseline_random_nonproj_edges)
  scores['baseline_random_proj'] = scorer.uuas(baseline_random_proj_edges)
  scores['projective'] = {}
  scores['projective']['edges'] = pmi_edges_proj
  scores['projective']['uuas'] = {}
  scores['non-projective'] = {}
  scores['non-projective']['edges'] = pmi_edges
  scores['non-projective']['uuas'] = {}

  for symmetrize_method in symmetrize_methods:
    scores['non-projective']['uuas'][symmetrize_method] = scorer.uuas(pmi_edges[symmetrize_method])
    scores['projective']['uuas'][symmetrize_method] = scorer.uuas(pmi_edges_proj[symmetrize_method])
  return scores

def print_tikz(tikz_filepath, predicted_edges, gold_edges, observation, label1='', label2=''):
  ''' Writes out a tikz dependency TeX file for comparing predicted_edges and gold_edges'''
  words = observation.sentence
  gold_edges_set = {tuple(sorted(x)) for x in gold_edges}

  gold_edge_label = {key : None for key in gold_edges_set}
  for i, _ in enumerate(observation.index):
    d, h = int(observation.index[i]), int(observation.head_indices[i])
    if (d-1, h-1) in gold_edges_set:
      gold_edge_label[(d-1,h-1)] = observation.governance_relations[i]
    elif (h-1, d-1) in gold_edges_set:
      gold_edge_label[(h-1,d-1)] = observation.governance_relations[i]

  predicted_edges_set = {tuple(sorted(x)) for x in predicted_edges}
  correct_edges = list(gold_edges_set.intersection(predicted_edges_set))
  incorrect_edges = list(predicted_edges_set.difference(gold_edges_set))
  num_correct = len(correct_edges)
  num_total = len(gold_edges)
  uuas = num_correct/float(num_total) if num_total != 0 else np.NaN
  # replace non-TeXsafe characters... add as needed
  tex_replace = { '$':'\$', '&':'+', '%':'\%', '~':'\textasciitilde', '#':'\#'}
  with open(tikz_filepath, 'a') as fout:
    string = "\\begin{dependency}\n\\begin{deptext}\n"
    string += "\\& ".join([tex_replace[x] if x in tex_replace else x for x in words]) + " \\\\" + '\n'
    string += "\\end{deptext}" + '\n'
    for i_index, j_index in gold_edge_label:
      string += f'\\depedge{{{i_index+1}}}{{{j_index+1}}}{{{gold_edge_label[(i_index, j_index)]}}}\n'
    for i_index, j_index in correct_edges:
      string += f'\\depedge[hide label, edge below, edge style={{blue, opacity=0.5}}]{{{i_index+1}}}{{{j_index+1}}}{{}}\n'
    for i_index, j_index in incorrect_edges:
      string += f'\\depedge[hide label, edge below, edge style={{red, opacity=0.5}}]{{{i_index+1}}}{{{j_index+1}}}{{}}\n'
    string += "\\node (R) at (\\matrixref.east) {{}};\n"
    string += f"\\node (R1) [right of = R] {{\\begin{{footnotesize}}{label1}\\end{{footnotesize}}}};"
    string += f"\\node (R2) at (R1.north) {{\\begin{{footnotesize}}{label2}\\end{{footnotesize}}}};"
    string += f"\\node (R3) at (R1.south) {{\\begin{{footnotesize}}{uuas:.2f}\\end{{footnotesize}}}};"
    string += f"\\end{{dependency}}\n"
    fout.write('\n\n')
    fout.write(string)

def get_padding(i, observations, threshold):
  '''
  to avoid short sentences on which XLNet performs badly as LM
  gets adjacent observations from PTB to add as padding,
  so total length is at least threshold ptb_tokens long
  (will truncate excessively long padding sentences)
  input: index and observations
  returns:
    prepadding: list of ptb tokens for before
    postpadding: list of ptb tokens for after
  '''
  j = i
  k = i
  pad_index_set = set()
  total_len = len(observations[i][0])
  while total_len < threshold:
    if j - 1 >= 0 and j - 1 not in pad_index_set:
      j -= 1
      pad_index_set.add(j)
      total_len += len(observations[j][0])
    if total_len >= threshold:
      break
    if k + 1 < len(observations) and k + 1 not in pad_index_set:
      k += 1
      pad_index_set.add(k)
      total_len += len(observations[k][0])
    else: raise ValueError(f'Not enough context to pad up to size {threshold}!')
  prepad_index_set = [x for x in sorted(pad_index_set) if x < i]
  postpad_index_set = [x for x in sorted(pad_index_set) if x > i]
  excessive = threshold # padding sentences longer than this will be truncated
  prepadding_observations = [observations[x] for x in prepad_index_set]
  prepadding = [i for x in [obs.sentence[:excessive] for obs in prepadding_observations] for i in x]
  postpadding_observations = [observations[x] for x in postpad_index_set]
  postpadding = [i for x in [obs.sentence[:excessive] for obs in postpadding_observations] for i in x]
  # print some explanation
  if pad_index_set != set():
    print(f'Using sentence(s) {sorted(pad_index_set)} as padding for sentence {i}.')
    print(f'|\tprepadding sentence lengths  : {[len(obs.sentence) for obs in prepadding_observations]}')
    for index, obs in zip(prepad_index_set, prepadding_observations):
      if len(obs.sentence) > excessive:
        print(f"|\t\t{index}: truncating at length {excessive}")
    print(f'|\tpostpadding sentence lengths : {[len(obs.sentence) for obs in postpadding_observations]}')
    for index, obs in zip(postpad_index_set, postpadding_observations):
      if len(obs.sentence) > excessive:
        print(f"|\t\ttruncating sentence {index} at length {excessive}")
  return prepadding, postpadding

def get_scores(
  observations, padlen=0, n_obs='all',
  save=False, verbose=False):
  '''get estimates get scores for n (default all) observations'''
  all_scores = []
  if n_obs == 'all':
    n_obs = len(observations)
  for i, obs in enumerate(tqdm(observations[:n_obs])):
    print(f'_______________\n--> Observation {i} of {n_obs}\n')
    if verbose:
      obs_df = pd.DataFrame(obs).T
      obs_df.columns = CONLL_COLS
      print(obs_df.loc[:, ['index', 'sentence', 'xpos_sentence', 'head_indices', 'governance_relations']],
            "\n", sep='')

    prepadding, postpadding = get_padding(i, observations, padlen)
    # get a pmi matrix and a pseudo-logprob for the sentence
    pmi_matrix, pseudo_loglik = MODEL.ptb_tokenlist_to_pmi_matrix(
        obs.sentence, add_special_tokens=True, verbose=verbose, # might want to toggle this verbosity during testing
        pad_left=prepadding, pad_right=postpadding)
    # calculate score
    scores = score_observation(obs, pmi_matrix)
    scores['pseudo_loglik'] = pseudo_loglik
    all_scores.append(scores)
    print(f"linear   {scores['baseline_linear']}")
    print(f"random   \n\tnon-proj   {scores['baseline_random_nonproj']}\n\tprojective {scores['baseline_random_proj']}")
    print(f"nonproj  {scores['non-projective']['uuas']}")
    print(f"proj     {scores['projective']['uuas']}\n")
  mean_linear = np.nanmean([scores['baseline_linear'] for scores in all_scores])
  mean_random_nonproj = np.nanmean([scores['baseline_random_nonproj'] for scores in all_scores])
  mean_random_proj = np.nanmean([scores['baseline_random_proj'] for scores in all_scores])
  mean_nonproj = {symmethod:np.nanmean([scores['non-projective']['uuas'][symmethod] for scores in all_scores]) for symmethod in ['sum', 'triu', 'tril', 'none']}
  mean_proj = {symmethod:np.nanmean([scores['projective']['uuas'][symmethod] for scores in all_scores]) for symmethod in ['sum', 'triu', 'tril', 'none']}
  print("=========\nmean uuas values\n")
  print(f'linear : {mean_linear:.3}')
  print(f'random :\n\tnon-proj   {mean_random_nonproj:.3}\n\tprojective {mean_random_proj:.3}')
  print(f'nonproj: { {k:round(v,2) for k, v in mean_nonproj.items()}}')
  print(f'proj   : { {k:round(v,2) for k, v in mean_proj.items()}}')
  return all_scores

if __name__ == '__main__':
  ARGP = ArgumentParser()
  ARGP.add_argument('--n_observations', default='all',
                    help='number of sentences to look at')
  ARGP.add_argument('--pmi_from_disk', nargs='?', const='pmi_matrices.npz',
                    help='to use saved matrices from disk (specify path/to/pmi_matrices.npz)') # TODO
  ARGP.add_argument('--model_spec', default='xlnet-base-cased',
                    help='''specify model (e.g. "xlnet-base-cased", "bert-large-cased"),
                    or path for offline''')
  ARGP.add_argument('--conllx_file', default='ptb3-wsj-data/ptb3-wsj-dev.conllx',
                    help='path/to/treebank.conllx: dependency file, in conllx format')
  ARGP.add_argument('--results_dir', default='results/',
                    help='specify path/to/results/directory/')
  # ARGP.add_argument('--save_matrices', action='store_true',
  #                   help='to save PMI matrices to disk.')
  ARGP.add_argument('--batch_size', default=32, type=int)
  ARGP.add_argument('--pad', default=0, type=int,
                    help='(int) pad sentences to be at least this long')
  CLI_ARGS = ARGP.parse_args()

  SPEC_STRING = str(CLI_ARGS.model_spec)

  N_OBS = CLI_ARGS.n_observations
  if N_OBS != 'all':
    N_OBS = int(N_OBS)

  NOW = datetime.now()
  DATE_SUFFIX = f'{NOW.year}-{NOW.month:02}-{NOW.day:02}-{NOW.hour:02}-{NOW.minute:02}'
  SPEC_SUFFIX = SPEC_STRING+'('+str(CLI_ARGS.n_observations)+')' if CLI_ARGS.n_observations != 'all' else SPEC_STRING
  SPEC_SUFFIX += '_pad'+str(CLI_ARGS.pad)
  SUFFIX = SPEC_SUFFIX + '_' + DATE_SUFFIX
  RESULTS_DIR = os.path.join(CLI_ARGS.results_dir, SUFFIX + '/')
  os.makedirs(RESULTS_DIR, exist_ok=True)
  print(f'RESULTS_DIR: {RESULTS_DIR}\n')

  print('Running pmi-accuracy.py with cli arguments:')
  with open(RESULTS_DIR+'spec.txt', mode='w') as specfile:
    for arg, value in sorted(vars(CLI_ARGS).items()):
      specfile.write(f"\t{arg}:\t{value}\n")
      print(f"\t{arg}:\t{value}")
    specfile.close()

  DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('Using device:', DEVICE)
  if DEVICE.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')

  # Instantiate the language model to use for getting estimates
  if CLI_ARGS.pmi_from_disk:
    MODEL_TYPE = 'load_from_disk'
  else:
    if CLI_ARGS.model_spec.startswith('xlnet'):
      MODEL_TYPE = 'xlnet'
      MODEL = languagemodel.XLNet(DEVICE, CLI_ARGS.model_spec, CLI_ARGS.batch_size)
    elif CLI_ARGS.model_spec.startswith('bert'):
      MODEL_TYPE = 'bert'
      MODEL = languagemodel.BERT(DEVICE, CLI_ARGS.model_spec, CLI_ARGS.batch_size)
    elif CLI_ARGS.model_spec.startswith('xlm'):
      MODEL_TYPE = 'xlm'
      MODEL = languagemodel.XLM(DEVICE, CLI_ARGS.model_spec, CLI_ARGS.batch_size)
    else:
      raise ValueError(f'Model spec string {CLI_ARGS.model_spec} not recognized.')

  # Columns of CONLL file
  CONLL_COLS = ['index',
                'sentence',
                'lemma_sentence',
                'upos_sentence',
                'xpos_sentence',
                'morph',
                'head_indices',
                'governance_relations',
                'secondary_relations',
                'extra_info']

  ObservationClass = namedtuple("Observation", CONLL_COLS)
  OBSERVATIONS = load_conll_dataset(CLI_ARGS.conllx_file, ObservationClass)

  SCORES = get_scores(OBSERVATIONS, padlen=CLI_ARGS.pad, n_obs=N_OBS, verbose=True)
  DF = pd.json_normalize(SCORES, sep='.')
  DF.to_csv(path_or_buf=RESULTS_DIR + 'scores_' + SUFFIX + '.csv',index_label='sentence_index')

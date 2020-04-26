from argparse import ArgumentParser
from collections import namedtuple

import main # to use functions accessing data

def print_tikz(tikz_filepath, predicted_edges, gold_edges, observation, label1='', label2=''):
  ''' Writes out a tikz dependency TeX file for comparing predicted_edges and gold_edges'''
  words = observation.sentence
  gold_edges_set = {tuple(sorted(x)) for x in gold_edges}

  gold_edge_label = {key : None for key in gold_edges_set}
  for i, _ in enumerate(observation.index):
    d, h = int(observation.index[i]), int(observation.head_indices[i])
    if (d-1, h-1) in gold_edges_set:
      gold_edge_label[(d-1, h-1)] = observation.governance_relations[i]
    elif (h-1, d-1) in gold_edges_set:
      gold_edge_label[(h-1, d-1)] = observation.governance_relations[i]

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


if __name__ == '__main__':
  ARGP = ArgumentParser()
  ARGP.add_argument('--sentence_index', default=0, type=int,
                    help='(int) sentence index')
  ARGP.add_argument('--conllx_file', default='ptb3-wsj-data/ptb3-wsj-dev.conllx',
                    help='path/to/treebank.conllx: dependency file, in conllx format')
  ARGP.add_argument('--input_file', default='scores.csv',
                    help='specify path/to/scores.csv')
  ARGP.add_argument('--output_file', default='out.tikz',
                    help='specify path/to/out.tikz')
  CLI_ARGS = ARGP.parse_args()

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
  OBSERVATIONS = main.load_conll_dataset(CLI_ARGS.conllx_file, ObservationClass)

  with CLI_ARGS.input_file as scores_csv:

  # NOT FINISHED
  
  print_tikz(
    CLI_ARGS.output_file, predicted_edges, gold_edges, 
    OBSERVATIONS[CLI_ARGS.sentence_index],
    label1=str(CLI_ARGS.sentence_index),label2='')
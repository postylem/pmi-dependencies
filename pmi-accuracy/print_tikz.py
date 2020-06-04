"""
Visualize dependency trees in TikZ.
"""
import glob
import os.path
import pandas as pd
import numpy as np

from argparse import ArgumentParser
from collections import namedtuple
from ast import literal_eval

import main  # to use functions accessing data


EXCLUDED_PUNCTUATION = ["", "'", "''", ",", ".", ";", "!", "?", ":", "``", "-LRB-", "-RRB-"]

def is_edge_to_ignore(edge, observation):
    is_d_punct = bool(observation.sentence[edge[1]-1] in EXCLUDED_PUNCTUATION)
    is_h_root = bool(edge[0] == 0)
    return is_d_punct or is_h_root


def make_tikz_string(
        predicted_edges, observation,
        label1='', label2=''):
    ''' Writes out a tikz dependency TeX file
    for comparing predicted_edges and gold_edges'''

    gold_edges_list = list(zip(list(map(int, observation.head_indices)),
                               list(map(int, observation.index)),
                               observation.governance_relations))
    gold_edge_to_label = {(e[0], e[1]): e[2] for e in gold_edges_list
                          if not is_edge_to_ignore(e, observation)}
    gold_edges_set = {tuple(sorted(e)) for e in gold_edge_to_label.keys()}

    # note converting to 1-indexing
    predicted_edges_set = {tuple(sorted((x[0]+1, x[1]+1))) for x in predicted_edges}
    correct_edges = list(gold_edges_set.intersection(predicted_edges_set))
    incorrect_edges = list(predicted_edges_set.difference(gold_edges_set))
    num_correct = len(correct_edges)
    num_total = len(gold_edges_set)
    uuas = num_correct/float(num_total) if num_total != 0 else np.NaN

    # replace non-TeXsafe characters... add as needed
    tex_replace = {'$': '\$', '&': '+', '%': '\%',
                   '~': '\textasciitilde', '#': '\#'}

    # make string
    string = "\\begin{dependency}\n\t\\begin{deptext}\n\t\t"
    string += "\\& ".join([tex_replace[x] if x in tex_replace
                           else x for x in observation.sentence]) + " \\\\" + '\n'
    string += "\t\\end{deptext}" + '\n'
    for i_index, j_index in gold_edge_to_label:
        string += f'\t\\depedge{{{i_index}}}{{{j_index}}}{{{gold_edge_to_label[(i_index, j_index)]}}}\n'
    for i_index, j_index in correct_edges:
        string += f'\t\\depedge[hide label, edge below, edge style={{-, blue, opacity=0.5}}]{{{i_index}}}{{{j_index}}}{{}}\n'
    for i_index, j_index in incorrect_edges:
        string += f'\t\\depedge[hide label, edge below, edge style={{-, red, opacity=0.5}}]{{{i_index}}}{{{j_index}}}{{}}\n'
    string += "\t\\node (R) at (\\matrixref.east) {{}};\n"
    string += f"\t\\node (R1) [right of = R] {{\\begin{{footnotesize}}{label1}\\end{{footnotesize}}}};\n"
    string += f"\t\\node (R2) at (R1.north) {{\\begin{{footnotesize}}{label2}\\end{{footnotesize}}}};\n"
    string += f"\t\\node (R3) at (R1.south) {{\\begin{{footnotesize}}{uuas*100:.0f}\\%\\end{{footnotesize}}}};\n"
    string += f"\\end{{dependency}}\n"
    return string


def write_tikz_files(
        outputdir, edges_df, sentence_indices,
        edge_type, output_prefix='', output_suffix=''):
    ''' writes TikZ string to outputdir,
    a seperate file for each sentence index'''
    for sentence_index in sentence_indices:
        predicted_edges = literal_eval(edges_df.at[sentence_index, edge_type])
        tikz_string = make_tikz_string(predicted_edges,
                                       OBSERVATIONS[sentence_index],
                                       label1=str(sentence_index),
                                       label2=output_suffix)
        tikzf = output_prefix + str(sentence_index) + output_suffix + ".tikz"
        tikzdir = os.path.join(outputdir, tikzf)
        print(f'writing tikz to {tikzdir}')
        with open(tikzdir, 'w') as fout:
            fout.write(f"% dependencies for {OUTPUTDIR}\n")
            fout.write(tikz_string)


if __name__ == '__main__':
    ARGP = ArgumentParser()
    ARGP.add_argument('--sentence_indices', type=int, nargs='+',
                      help='''sentence indices to plot dependencies for.
                              enter integer(s)''')
    ARGP.add_argument('--input_file',
                      default='scores.csv',
                      help='specify path/to/scores.csv')
    ARGP.add_argument('--conllx_file',
                      default='ptb3-wsj-data/ptb3-wsj-dev.conllx',
                      help='path/to/treebank.conllx: dependency file')
    ARGP.add_argument('--output_prefix',
                      default='',
                      help='specify output [prefix]sentence_index.tikz')
    ARGP.add_argument('--edge_types', type=str, default='projective.edges.sum',
                      nargs='+',
                      help="""Edge type to plot against the gold.
                              Chose any subset of
                              ['projective.edges.sum',
                              'projective.edges.triu',
                              'projective.edges.tril',
                              'projective.edges.none',
                              'nonproj.edges.sum',
                              'nonproj.edges.triu',
                              'nonproj.edges.tril',
                              'nonproj.edges.none'],
                              or enter 'all' for all.""")
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
    OBSERVATIONS = main.load_conll_dataset(CLI_ARGS.conllx_file,
                                           ObservationClass)
    OUTPUTDIR = os.path.dirname(CLI_ARGS.input_file)
    EDGES_DF = pd.read_csv(CLI_ARGS.input_file)

    if CLI_ARGS.edge_types == ['all']:
        print(CLI_ARGS.edge_types)
        CLI_ARGS.edge_types = [
            'projective.edges.sum', 'projective.edges.triu',
            'projective.edges.tril', 'projective.edges.none',
            'nonproj.edges.sum', 'nonproj.edges.triu',
            'nonproj.edges.tril', 'nonproj.edges.none'
            ]
    for edge_type in CLI_ARGS.edge_types:
        edgetype = edge_type.split(".")
        label = f'{edgetype[2]}.{edgetype[0]}'
        write_tikz_files(OUTPUTDIR, EDGES_DF,
                         CLI_ARGS.sentence_indices, edge_type,
                         output_prefix=CLI_ARGS.output_prefix,
                         output_suffix=label)

    TEX_FILEPATH = os.path.join(OUTPUTDIR, 'dependencies.tex')
    with open(TEX_FILEPATH, mode='w') as tex_file:
        print(f'writing TeX to {TEX_FILEPATH}')
        tex_file.write(
            "\\documentclass[tikz]{standalone}\n"
            "\\usepackage{tikz,tikz-dependency}\n"
            "\\pgfkeys{%\n/depgraph/edge unit distance=.75ex,%\n"
            "%/depgraph/edge horizontal padding=2,%\n"
            "/depgraph/reserved/edge style/.style = {\n->, % arrow properties\n"
            "semithick, solid, line cap=round, % line properties\n"
            "rounded corners=2, % make corners round\n},%\n"
            "/depgraph/reserved/label style/.style = {%\n"
            "% anchor = mid, draw, solid, black, rotate = 0,"
            "rounded corners = 2pt,%\nscale = .5,%\ntext height = 1.5ex,"
            "text depth = 0.25ex, % needed to center text vertically\n"
            "inner sep=.2ex,%\nouter sep = 0pt,%\ntext = black,%\n"
            "fill = white, %opacity = 0, text opacity = 0 "
            "% uncomment to hide all labels\n},%\n}\n"
            "\\begin{document}\n\n% % Put tikz dependencies here, like\n"
            )
        tex_file.write(f"% dependencies for {OUTPUTDIR}\n")
        TIKZFILES = glob.glob(os.path.join(OUTPUTDIR, '*.tikz'))
        TIKZFILES = [os.path.basename(x) for x in TIKZFILES]
        for tikzfile in sorted(TIKZFILES):
            tex_file.write(f"\\input{{{tikzfile}}}\n")
        tex_file.write("\n\\end{document}")


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

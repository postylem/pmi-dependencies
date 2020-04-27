#!/bin/bash
# Prints some sample tikz files from the specified scores file.  Change which here.
# SCORES_FILE='path/to/scores.csv'
SCORES_FILE=${1?Error: no scores filepath given}

python pmi-accuracy/print_tikz.py --sentence_indices 166 1696 --conllx_file ptb3-wsj-data/ptb3-wsj-dev.conllx --input_file $SCORES_FILE --edge_types nonproj.edges.sum 

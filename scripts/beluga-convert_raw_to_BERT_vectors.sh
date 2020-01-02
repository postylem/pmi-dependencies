#!/bin/bash
#
#
# convert conllx splits to plaintext line-per-sentence
# Use with input filepath prefix like `path/to/ptb-wsj` and bert model `large`

PATH_PREFIX=${1?Error: no input filepath given} # can use path relative to project
BERT_MODEL=${2?Error: specify BERT model: base or large} # `base` or `large`

for split in train dev test; do
    echo Getting vectors for $split split...
    python3 ~/wdir-morgan/proj4/hewitt-repo/scripts/convert_raw_to_bert.py $PATH_PREFIX-${split}.txt $PATH_PREFIX-${split}-BERT$BERT_MODEL.vectors.hdf5 $BERT_MODEL
done

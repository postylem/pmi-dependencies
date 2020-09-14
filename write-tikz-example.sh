#!/bin/bash
modelspecs=(bart-large bert-base-cased bert-large-cased distilbert gpt2 w2v xlm xlnet-base xlnet-large)
for modelspec in "${modelspecs[@]}"; do
  python pmi_accuracy/print_tikz.py --sentence_indices 442 --input_file results-clean/contextual_embedding_models/${modelspec}*/loaded*/scores*.csv --info ${modelspec}-abs
done

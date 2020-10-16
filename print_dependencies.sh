#!/bin/bash

## CEMs
#python pmi_accuracy/print_tikz.py --sentence_indices 78 381 405 442 1081 1352 --input_file results-clean/contextual_embedding_models/bart*/loaded*/scores*.csv --output_dir tikz/con*/ --info Bart
#python pmi_accuracy/print_tikz.py --sentence_indices 78 381 405 442 1081 1352 --input_file results-clean/contextual_embedding_models/bert-base-cased*/loaded*/scores*.csv --output_dir tikz/con*/ --info BERT-base
#python pmi_accuracy/print_tikz.py --sentence_indices 78 381 405 442 1081 1352 --input_file results-clean/contextual_embedding_models/bert-large-cased*/loaded*/scores*.csv --output_dir tikz/con*/ --info BERT-large
#python pmi_accuracy/print_tikz.py --sentence_indices 78 381 405 442 1081 1352 --input_file results-clean/contextual_embedding_models/distilbert*/loaded*/scores*.csv --output_dir tikz/con*/ --info DistilBERT
#python pmi_accuracy/print_tikz.py --sentence_indices 78 381 405 442 1081 1352 --input_file results-clean/contextual_embedding_models/gpt2*/loaded*/scores*.csv --output_dir tikz/con*/ --info GPT2
#python pmi_accuracy/print_tikz.py --sentence_indices 78 381 405 442 1081 1352 --input_file results-clean/contextual_embedding_models/w2v*/scores*.csv --output_dir tikz/con*/ --info W2V-signed
#python pmi_accuracy/print_tikz.py --sentence_indices 78 381 405 442 1081 1352 --input_file results-clean/contextual_embedding_models/xlm*/loaded*/scores*.csv --output_dir tikz/con*/ --info XLM
#python pmi_accuracy/print_tikz.py --sentence_indices 78 381 405 442 1081 1352 --input_file results-clean/contextual_embedding_models/xlnet-base*/loaded*/scores*.csv --output_dir tikz/con*/ --info XLNet-base
#python pmi_accuracy/print_tikz.py --sentence_indices 78 381 405 442 1081 1352 --input_file results-clean/contextual_embedding_models/xlnet-large*/loaded*/scores*.csv --output_dir tikz/con*/ --info XLNet-large

## LSTM ONLSTM ONLSTM-SYD
#python pmi_accuracy/print_tikz.py --sentence_indices 78 381 405 442 1081 1352 --input_file results-clean/lstms/*=lstm-abs*/scores*.csv --output_dir tikz/lstms/ --info LSTM
#python pmi_accuracy/print_tikz.py --sentence_indices 78 381 405 442 1081 1352 --input_file results-clean/lstms/*=onlstm-abs*/scores*.csv --output_dir tikz/lstms/ --info ONLSTM
#python pmi_accuracy/print_tikz.py --sentence_indices 78 381 405 442 1081 1352 --input_file results-clean/lstms/*=onlstm_syd-abs*/scores*.csv --output_dir tikz/lstms/ --info ONLSTM-SYD

## POS simple/IB probe
#python pmi_accuracy/print_tikz.py --sentence_indices 78 381 405 442 1081 1352 --input_file results-clean/pos-cpmi/IB*/*xpos_bert-base*/scores*.csv --output_dir tikz/pos-cpmi/ --info POS-IB-BERT-base
#python pmi_accuracy/print_tikz.py --sentence_indices 78 381 405 442 1081 1352 --input_file results-clean/pos-cpmi/IB*/*xpos_bert-large*/scores*.csv --output_dir tikz/pos-cpmi/ --info POS-IB-BERT-large
#python pmi_accuracy/print_tikz.py --sentence_indices 78 381 405 442 1081 1352 --input_file results-clean/pos-cpmi/IB*/*xpos_xlnet-base*/scores*.csv --output_dir tikz/pos-cpmi/ --info POS-IB-XLNet-base
#python pmi_accuracy/print_tikz.py --sentence_indices 78 381 405 442 1081 1352 --input_file results-clean/pos-cpmi/IB*/*xpos_xlnet-large*/scores*.csv --output_dir tikz/pos-cpmi/ --info POS-IB-XLNet-base
#python pmi_accuracy/print_tikz.py --sentence_indices 78 381 405 442 1081 1352 --input_file results-clean/pos-cpmi/simple*/xpos_bert-base*/scores*.csv --output_dir tikz/pos-cpmi/ --info POS-simple-BERT-base
#python pmi_accuracy/print_tikz.py --sentence_indices 78 381 405 442 1081 1352 --input_file results-clean/pos-cpmi/simple*/xpos_bert-large*/scores*.csv --output_dir tikz/pos-cpmi/ --info POS-simple-BERT-large
#python pmi_accuracy/print_tikz.py --sentence_indices 78 381 405 442 1081 1352 --input_file results-clean/pos-cpmi/simple*/xpos_xlnet-base*/scores*.csv --output_dir tikz/pos-cpmi/ --info POS-simple-XLNet-base
#python pmi_accuracy/print_tikz.py --sentence_indices 78 381 405 442 1081 1352 --input_file results-clean/pos-cpmi/simple*/xpos_xlnet-large*/scores*.csv --output_dir tikz/pos-cpmi/ --info POS-simple-XLNet-large

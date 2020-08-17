#!/bin/bash

# Checkpointed bert-base-uncased 
#for  n in 10 50 100 500 1000 1500; do
#  python pmi-accuracy/main.py --model_spec bert-base-uncased --model_path ~/bert_base_ckpt/model_steps_${n}000.pt --pad 30 --batch_size 16 --save_npz > bert-ckpt${n}k.out
#done

# # Saving matrices for all models (to be placed in results-clean/)
#python pmi-accuracy/main.py --save_npz --model_spec bert-large-cased --pad 60 --batch_size 4 > bert-large-cased-pad60.out
#python pmi-accuracy/main.py --save_npz --model_spec bert-base-cased --pad 30 --batch_size 16 > bert-base-cased-pad30.out
#python pmi-accuracy/main.py --save_npz --model_spec bert-large-uncased --pad 30 --batch_size 10 > bert-large-uncased-pad30.out
#python pmi-accuracy/main.py --save_npz --model_spec bert-base-uncased --pad 30 --batch_size 10 > bert-base-uncased-pad30.out
#python pmi-accuracy/main.py --save_npz --model_spec xlnet-large-cased --pad 30 --batch_size 4 > xlnet-large-cased-pad30.out
#python pmi-accuracy/main.py --save_npz --model_spec xlnet-base-cased --pad 30 --batch_size 10 > xlnet-base-cased-pad30.out
#python pmi-accuracy/main.py --save_npz --model_spec xlm-mlm-en-2048 --pad 60 --batch_size 4 > xlm-mlm-en-2048-pad60.out
#python pmi-accuracy/main.py --save_npz --model_spec bart-large --pad 60 --batch_size 4 > bart-large-pad60.out
#python pmi-accuracy/main.py --save_npz --model_spec distilbert-base-cased --pad 60 --batch_size 24 > distilbert-base-cased-pad60.out
#python pmi-accuracy/main.py --save_npz --model_spec w2v --model_path gensim_w2v.txt --pad 0 > w2v.out

# # Getting absolute value version (assuming only the just calculated scores are in results-clean/)

#for dir in results-clean/*; do
#  python pmi-accuracy/main.py --model_spec load_npz --model_path "$dir" --absolute_value > `basename "$dir"`.out
#done

# just checking,  on test set rather than dev
#python pmi-accuracy/main.py --save_npz --model_spec bert-large-cased --pad 60 --batch_size 6 --conllx_file 'ptb3-wsj-data/ptb3-wsj-test.conllx' > bert-large-cased-pad60-test.out 2> bert-large-cased-pad60-test.err

#-------- POS PROBE ---------#

# train probe
#python pmi-accuracy/pos_probe.py --model_spec bert-base-cased --pos_set_type upos
#python pmi-accuracy/pos_probe.py --model_spec bert-large-cased --pos_set_type upos

# evaluate linear probe bert
#python pmi-accuracy/main.py --save_npz --model_spec bert-base-cased --pad 30 --batch_size 16 --probe_state_dict probe-results/bert-base-cased_20.07.15-20.06/probe.state_dict --pos_set_type upos > upos-bert-base-cased.out 2> upos-base.err
#python pmi-accuracy/main.py --save_npz --model_spec bert-large-cased --pad 60 --batch_size 4 --probe_state_dict probe-results/bert-large-cased_20.07.15-20.28/probe.state_dict --pos_set_type upos > upos-bert-large-cased.out 2> upos-large.err

# bert get abs value version also
#python pmi-accuracy/main.py --model_spec load_npz --model_path results/upos_bert-base-cased_pad30_2020-07-15-21-55 --absolute_value --results_dir results/upos_bert-base-cased_pad30_2020-07-15-21-55/ > upos-base-abs.out
#python pmi-accuracy/main.py --model_spec load_npz --model_path results/upos_bert-large-cased_pad60_2020-07-15-22-52 --absolute_value --results_dir results/upos_bert-large-cased_pad60_2020-07-15-22-52/ > upos-large-abs.out

# train probe xlnet
#python pmi-accuracy/pos_probe.py --model_spec xlnet-base-cased  --pos_set_type upos
#python pmi-accuracy/pos_probe.py --model_spec xlnet-large-cased --pos_set_type upos

# evaluate linear probe xlnet
#python pmi-accuracy/main.py --save_npz --model_spec xlnet-base-cased --pad 30 --batch_size 32 --probe_state_dict probe-results/upos_xlnet-base-cased*/probe.state_dict --pos_set_type upos > upos-base.out 2> upos-base.err
#python pmi-accuracy/main.py --save_npz --model_spec xlnet-large-cased --pad 30 --batch_size 16 --probe_state_dict probe-results/upos_xlnet-large-cased*/probe.state_dict --pos_set_type upos > upos-large.out 2> upos-large.err

# xlnet get abs value version also
#python pmi-accuracy/main.py --model_spec load_npz --model_path results/upos_xlnet-base-cased* --absolute_value --results_dir results/upos_xlnet-base-cased*/ > upos-base-abs.out
#python pmi-accuracy/main.py --model_spec load_npz --model_path results/upos_xlnet-large-cased* --absolute_value --results_dir results/upos_xlnet-large-cased*/ > upos-large-abs.out

# Running Li's default
#python pmi-accuracy/pos_probe.py --model_spec bert-base-cased --beta 1e-5 --optimizer adam --lr 0.001 --weight_decay 0.0001
#python pmi-accuracy/pos_probe.py --model_spec bert-base-cased --bottleneck --beta 1e-5 --optimizer adam --lr 0.001 --weight_decay 0.0001
#python pmi-accuracy/pos_probe.py --model_spec bert-large-cased --bottleneck --beta 1e-5 --optimizer adam --lr 0.001 --weight_decay 0.0001
#python pmi-accuracy/pos_probe.py --model_spec xlnet-base-cased --bottleneck --beta 1e-5 --optimizer adam --lr 0.001 --weight_decay 0.0001
#python pmi-accuracy/pos_probe.py --model_spec xlnet-large-cased --bottleneck --beta 1e-5 --optimizer adam --lr 0.001 --weight_decay 0.0001

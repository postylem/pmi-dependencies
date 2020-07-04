#!/bin/bash

# keep track of things I've run

#python pmi-accuracy/main.py --model_spec distilbert-base-cased --pad 100 --batch_size 8 > distilbert-pad100.out 2> distilbert-pad100.err
#python pmi-accuracy/main.py --model_spec distilbert-base-cased --pad 60 --batch_size 24 > distilbert-pad60.out 2> distilbert-pad60.err
#python pmi-accuracy/main.py --model_spec bert-base-cased --pad 60 --batch_size 16 > bert-base-cased-pad60.out 2> bert-base-cased-pad60.err
#python pmi-accuracy/main.py --model_spec bert-large-cased --pad 60 --batch_size 4 > bert-large-cased-pad60.out 2> bert-large-cased-pad60.err
#python pmi-accuracy/main.py --model_spec xlm-mlm-en-2048 --pad 60 --batch_size 4 > xlm-pad60.out 2> xlm-pad60.err
#python pmi-accuracy/main.py --model_spec bert-base-cased --pad 30 --batch_size 16 > bert-base-cased-pad30.out 2> bert-base-cased-pad30.err
#python pmi-accuracy/main.py --model_spec bert-large-cased --pad 30 --batch_size 10 > bert-large-cased-pad30.out 2> bert-large-cased-pad30.err
#python pmi-accuracy/main.py --model_spec xlnet-base-cased --pad 30 --batch_size 16 > xlnet-base-pad30.out 2> xlnet-base-pad30.err
#python pmi-accuracy/main.py --model_spec xlnet-large-cased --pad 30 --batch_size 10 > xlnet-large-pad30.out 2> xlnet-large-pad30.err
#python pmi-accuracy/main.py --model_spec xlnet-large-cased --pad 30 --batch_size 10 > xlnet-large-pad30.out 2> xlnet-large-pad30.err
#python pmi-accuracy/main.py --model_spec w2v --pad 0 > w2v.out 2> w2v.err
#python pmi-accuracy/main.py --model_spec w2v --pad 0 --model_path gensim_w2v.txt > gensim_w2v.out 2> gensim_w2v.err

# # Running bert-base-uncased to compare with checkpoints:

#python pmi-accuracy/main.py --model_spec bert-base-uncased --pad 60 --batch_size 16 --n_observations 500 > bert-base-uncased-pad60.out 2> bert-base-uncased-pad60.err
#python pmi-accuracy/main.py --model_spec bert-base-uncased --model_path ~/bert_base_ckpt/model_steps_1500000.pt --pad 60 --batch_size 16 --n_observations 500 > bert-ckpt1500k.out 2> bert-ckpt1500k.err
#python pmi-accuracy/main.py --model_spec bert-base-uncased --model_path ~/bert_base_ckpt/model_steps_1000000.pt --pad 60 --batch_size 16 --n_observations 500 > bert-ckpt1000k.out 2> bert-ckpt1000k.err
#python pmi-accuracy/main.py --model_spec bert-base-uncased --model_path ~/bert_base_ckpt/model_steps_500000.pt --pad 60 --batch_size 16 --n_observations 500 > bert-ckpt500k.out 2> bert-ckpt500k.err
#python pmi-accuracy/main.py --model_spec bert-base-uncased --model_path ~/bert_base_ckpt/model_steps_100000.pt --pad 60 --batch_size 16 --n_observations 500 > bert-ckpt100k.out 2> bert-ckpt100k.err
#python pmi-accuracy/main.py --model_spec bert-base-uncased --model_path ~/bert_base_ckpt/model_steps_50000.pt --pad 60 --batch_size 16 --n_observations 500 > bert-ckpt50k.out 2> bert-ckpt50k.err
#python pmi-accuracy/main.py --model_spec bert-base-uncased --model_path ~/bert_base_ckpt/model_steps_10000.pt --pad 60 --batch_size 16 --n_observations 500 > bert-ckpt10k.out 2> bert-ckpt10k.err

python pmi-accuracy/main.py --save_npz --model_spec bert-large-cased --pad 60 --batch_size 4 > bert-large-cased-pad60.out 2> bert-large-cased-pad60.err
python pmi-accuracy/main.py --save_npz --model_spec bert-base-cased --pad 30 --batch_size 10 > bert-base-cased-pad30.out 2> bert-base-cased-pad30.err
python pmi-accuracy/main.py --save_npz --model_spec bert-large-uncased --pad 30 --batch_size 10 > bert-large-uncased-pad30.out 2> bert-large-uncased-pad30.err
python pmi-accuracy/main.py --save_npz --model_spec bert-base-uncased --pad 30 --batch_size 10 > bert-base-uncased-pad30.out 2> bert-base-uncased-pad30.err
python pmi-accuracy/main.py --save_npz --model_spec xlnet-large-cased --pad 30 --batch_size 10 > xlnet-large-cased-pad30.out 2> xlnet-large-cased-pad30.err
python pmi-accuracy/main.py --save_npz --model_spec xlnet-base-cased --pad 30 --batch_size 10 > xlnet-base-cased-pad30.out 2> xlnet-base-cased-pad30.err
python pmi-accuracy/main.py --save_npz --model_spec xlm-mlm-en-2048 --pad 60 --batch_size 4 > xlm-mlm-en-2048-pad60.out 2> xlm-mlm-en-2048-pad60.err
python pmi-accuracy/main.py --save_npz --model_spec bart-large --pad 60 --batch_size 4 > bart-large-pad60.out 2> bart-large-pad60.err
python pmi-accuracy/main.py --save_npz --model_spec distilbert-base-cased --pad 60 --batch_size 24 > distilbert-base-cased-pad60.out 2> distilbert-base-cased-pad60.err
python pmi-accuracy/main.py --save_npz --model_spec w2v --model_path gensim_w2v.txt --pad 0 > w2v.out 2> w2v.err

echo "all done!"
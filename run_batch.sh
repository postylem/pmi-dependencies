#!/bin/bash

# keep track of things I've run, for pseudologlik by commenting them out for now

#python pmi-accuracy/main.py --model_spec bert-base-cased --pad 60 --batch_size 16 > bert-base-cased-pad60.out 2> bert-base-cased-pad60.err
python pmi-accuracy/main.py --model_spec bert-large-cased --pad 60 --batch_size 4 > bert-large-cased-pad60.out 2> bert-large-cased-pad60.err
#python pmi-accuracy/main.py --model_spec xlm-mlm-en-2048 --pad 60 --batch_size 4 > xlm-pad60.out 2> xlm-pad60.err
#python pmi-accuracy/main.py --model_spec bert-base-cased --pad 30 --batch_size 16 > bert-base-cased-pad30.out 2> bert-base-cased-pad30.err
#python pmi-accuracy/main.py --model_spec bert-large-cased --pad 30 --batch_size 10 > bert-large-cased-pad30.out 2> bert-large-cased-pad30.err
python pmi-accuracy/main.py --model_spec xlnet-base-cased --pad 30 --batch_size 16 > xlnet-base-pad30.out 2> xlnet-base-pad30.err
#python pmi-accuracy/main.py --model_spec xlnet-large-cased --pad 30 --batch_size 10 > xlnet-large-pad30.out 2> xlnet-large-pad30.err
#python pmi-accuracy/main.py --model_spec xlnet-large-cased --pad 30 --batch_size 10 > xlnet-large-pad30.out 2> xlnet-large-pad30.err


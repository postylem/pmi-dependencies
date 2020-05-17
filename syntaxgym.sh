#!/bin/bash
# get PMI estimates for all sentences in each of the SyntaxGym test suites, from various models

# models
modelArray=("xlnet-base-cased" "bert-large-cased" "xlm-mlm-en-2048" "bart-large" "distilbert-base-cased")
for model in ${modelArray[*]};do
  python pmi-accuracy/txt_to_pmi.py --txt SyntaxGym_test_suites/txt/ --model_spec "$model" > "SG-${model}.out"
done

python pmi-accuracy/txt_to_pmi.py --txt SyntaxGym_test_suites/txt/ --model_spec w2v --model_path gensim_w2v.txt > SG-gensim_w2v.out


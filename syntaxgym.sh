#!/bin/bash

# models
modelArray=("xlnet-base-cased" "")
for model in ${modelArray[*]};do
  python pmi-accuracy/txt_to_pmi.py --txt SyntaxGym_test_suites/txt/number_src.txt --model_spec xlnet-base-cased
done


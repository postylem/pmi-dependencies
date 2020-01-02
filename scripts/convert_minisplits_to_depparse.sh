#!/bin/bash
#
# Copyright 2017 The Board of Trustees of The Leland Stanford Junior University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#
# Author: Peng Qi
# Modified by John Hewitt
# Modified by Jacob Hoover

# location of the Penn Treebank WSJ parsed .mrg files (set your own)
WSJ='/Users/j/McGill/PhD-miasma/syntactic-embeddings/code/Treebank3/PARSED/MRG/WSJ'
# location of the stanford-corenlp (set your own)
CORENLP='/Users/j/McGill/PhD-miasma/syntactic-embeddings/code/stanford-corenlp-full-2018-10-05'

# The standard train def test split of WSJ
# train sections: 2-21
for i in `seq -w 19 21`; do
        cat ${WSJ}/$i/*.MRG
done > ptb3-wsj-train.trees

# dev section: 22
for i in 22; do
        cat ${WSJ}/$i/*.MRG
done > ptb3-wsj-dev.trees

# test section: 23
for i in 23; do
        cat ${WSJ}/$i/*.MRG
done > ptb3-wsj-test.trees

for split in train dev test; do
    echo Converting $split split...
    # be sure to select conllx or conllu as desired: 
    # uncomment for conllu (very slow):
    # java -cp "${CORENLP}/*" -mx1g edu.stanford.nlp.trees.ud.UniversalDependenciesConverter -treeFile ptb3-wsj-${split}.trees > ptb3-wsj-${split}.conllu
    # uncomment for conllx:
    java -cp "${CORENLP}/*" -mx1g edu.stanford.nlp.trees.EnglishGrammaticalStructure -treeFile ptb3-wsj-${split}.trees -checkConnected -basic -keepPunct -conllx > ptb3-wsj-${split}.conllx
done

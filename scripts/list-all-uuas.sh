#!/bin/bash
# simple example:
# for uuas in results/*+16*/dev.uuas; do echo $uuas; cat $uuas; done

PROBE_TYPE=${1?linear or deep}
OUT=2layer-$PROBE_TYPE-results.csv

for i in {00..23}; do
  for j in {00..23}; do
    cat results/BERT-2layer-$PROBE_TYPE/BERT-disk-$i+$j-*/dev.uuas | tr '\n' ',' >> $OUT
  done
  printf "\n" >> $OUT
done

#!/bin/bash
#
#
# run a bevy of probes

for i in 8 9 10 11; do
	echo "probing bert-base layer $i"
	python hewitt-repo/structural-probes/run_experiment.py miniptb3-wsj-data/miniptb-BERTbase$i-deep.yaml
done
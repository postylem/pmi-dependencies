#!/bin/bash
#SBATCH --account=def-msonde1
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=32000M              # memory (per node)
#SBATCH --time=0-04:00            # time (DD-HH:MM)
#SBATCH --job-name=write_bert_layers
#SBATCH --output=%x-%j.out
###########################
set -ex

module load python/3.7 cuda cudnn
SOURCEDIR=~/wdir-morgan/proj4


# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index -r $SOURCEDIR/requirements.txt

# convert to BERT vectors
# ~/wdir-morgan/proj4/scripts/convert_raw_to_BERT_vectors.sh ~/wdir-morgan/proj4/ptb3-wsj-data/ptb3-wsj large

PATH_PREFIX=~/wdir-morgan/proj4/ptb3-wsj-data/ptb3-wsj
BERT_MODEL=large
BERT_PATH=~/wdir-morgan/proj4/BERT/

for split in train dev test; do
    echo Getting vectors for $split split...
    python ~/wdir-morgan/proj4/hewitt-repo/scripts/convert_raw_to_bert.py $PATH_PREFIX-${split}.txt $PATH_PREFIX-${split}-BERT$BERT_MODEL.vectors.hdf5 $BERT_MODEL --bert_path $BERT_PATH
done
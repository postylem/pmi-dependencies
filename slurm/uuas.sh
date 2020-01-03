#!/bin/bash
#SBATCH --account=def-msonde1
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=15G                # memory (per node)
#SBATCH --time=0-10:00            # time (DD-HH:MM)
#SBATCH --job-name=uuas
#SBATCH --output=%x-%j.out
###########################
# For verbose version
set -ex

module load python/3.7 cuda cudnn
SOURCEDIR=~/wdir-morgan/xlnet-pmi

# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index torch
pip install --no-index pytorch-transformers
pip install --no-index -r $SOURCEDIR/requirements.txt

PATH_PREFIX=~/wdir-morgan/xlnet-pmi
BATCH_SIZE=${1?Error: no batch size specified}

START=`date +%s`
python $PATH_PREFIX/pmi-accuracy/pmi-accuracy.py --batch-size $BATCH_SIZE --results-dir $PATH_PREFIX/results-cluster/
echo "Duration: $((($(date +%s)-$START)/60)) minutes"

#!/bin/bash
#SBATCH --account=def-msonde1
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=32000M              # memory (per node)
#SBATCH --time=0-04:00            # time (DD-HH:MM)
#SBATCH --job-name=probe
#SBATCH --output=%x-%j.out
###########################
set -ex

module load python/3.7 cuda cudnn
SOURCEDIR=~/wdir-morgan/proj4

# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index -r $SOURCEDIR/requirements.txt

CONFIG_YAML=${1?Error: no config file given}
PATH_PREFIX=~/wdir-morgan/proj4

START=`date +%s`
python $PATH_PREFIX/hewitt-repo/structural-probes/run_experiment.py $PATH_PREFIX/configs/$CONFIG_YAML
echo "Duration: $((($(date +%s)-$START)/60)) minutes"

#!/bin/bash
# CPU job with the conda base env (pandas/matplotlib/scipy): sbatch base_env.sbatch <script.py> [args]
#SBATCH -J shape-rl-analysis
#SBATCH -p test
#SBATCH --account=wattenberg_lab
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 0-00:20
module load python
source activate $LABDIR/conda_envs/base
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python -I "$@"

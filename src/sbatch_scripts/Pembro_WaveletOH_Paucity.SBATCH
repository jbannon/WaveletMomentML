#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --time=120:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=Pembro_WaveletOH_Paucity
#SBATCH --output=./slurmouts/Pembro_WaveletOH_Paucity_%j.out
#SBATCH --mail-type=TIME_LIMIT,BEGIN,END,FAIL
#SBATCH --mail-user=jjb509@nyu.edu

module purge


singularity exec --nv \
	    --overlay /scratch/jjb509/WaveletMomentML/src/wavelet_moment_overlay.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python paucity.py -config ./configs/Pembro_Wavelet_OH.yaml"

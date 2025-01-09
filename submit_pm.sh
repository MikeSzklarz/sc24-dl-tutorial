#!/bin/bash

#SBATCH --job-name=vit-era5_%j                      # Job name
#SBATCH --output=logs/job_%j.txt                    # Output log
#SBATCH --ntasks=1                                  # Number of tasks
#SBATCH --mem=65536                                 # Memory (64 GB)
#SBATCH --time=30-00:00:00                          # Job time limit
#SBATCH --partition=waccamaw                        # Partition to use
#SBATCH --exclusive                                 # Exclusive node allocation
#SBATCH --exclude=waccamaw02,waccamaw03,waccamaw04  # Exclude specific nodes


DATADIR=${PWD}/data
LOGDIR=${PWD}/logs
mkdir -p ${LOGDIR}
args="${@}"

# Load the environment
# Uncomment below if erroring
# source /mnt/cidstore1/software/debian12/anaconda3/etc/profile.d/conda.sh 2>> logs/vit-era5_${SLURM_JOB_ID}_error.txt
# conda activate nersc24 2>> logs/vit-era5_${SLURM_JOB_ID}_error.txt

# No Errors dont need extra file
source /mnt/cidstore1/software/debian12/anaconda3/etc/profile.d/conda.sh
conda activate nersc24

export FI_MR_CACHE_MONITOR=userfaultfd
export HDF5_USE_FILE_LOCKING=FALSE

# Profiling
if [ "${ENABLE_PROFILING:-0}" -eq 1 ]; then
    echo "Enabling profiling..."

    # Check for memory profiling flag
    if [ "${ENABLE_MEMORY_CAPTURE:-0}" -eq 1 ]; then
        echo "Enabling memory capture (mem, osrt)..."
        MEM_FLAGS=",osrt"  # Additional flags for memory profiling
    else
        MEM_FLAGS=""
    fi

    NSYS_ARGS="--trace=cuda,cublas,nvtx${MEM_FLAGS} --kill none -c cudaProfilerApi -f true"
    NSYS_OUTPUT=${LOGDIR}/${PROFILE_OUTPUT:-"profile"}
    export PROFILE_CMD="nsys profile $NSYS_ARGS -o $NSYS_OUTPUT"
fi

export MASTER_ADDR=$(hostname)

# Having multiple gpus available when only one is avaiable causes issues
export CUDA_VISIBLE_DEVICES=0

# Debugging mode
set -x

# Run the training script directly
srun -u bash -c "
    source export_DDP_vars.sh
    ${PROFILE_CMD} python train.py ${args}
"

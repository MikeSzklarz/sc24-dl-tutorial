#!/bin/bash 

#SBATCH --job-name=vit-era5_mn_%j                # Job name
#SBATCH --output=logs/slurm/job_%j.txt           # Output log MAKE SURE DIR EXISTS

#SBATCH --nodes=1                                # Number of nodes
#SBATCH --ntasks-per-node=1                      # Number of tasks per node

#SBATCH --mem=65GB                               # Memory (65 GB)
#SBATCH --time=00-00:30:00                       # Job time limit
#SBATCH --partition=waccamaw                     # Partition to use
#SBATCH --exclusive                              # Exclusive node allocation
#SBATCH --exclude=waccamaw03,waccamaw04          # Exclude specific nodes

DATADIR=${PWD}/data
LOGDIR=${PWD}/logs
mkdir -p ${LOGDIR}
args="${@}"

export FI_MR_CACHE_MONITOR=userfaultfd
export HDF5_USE_FILE_LOCKING=FALSE

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
# export LOGLEVEL=INFO

# export NCCL_DEBUG=INFO          # Uncomment to debug NCCL 
export NCCL_SOCKET_IFNAME=eno8303 # If not set, NCCL uses the wrong network interface

# Load the environment
# Uncomment below if erroring
# source /mnt/cidstore1/software/debian12/anaconda3/etc/profile.d/conda.sh 2>> logs/vit-era5_${SLURM_JOB_ID}_error.txt
# conda activate nersc24 2>> logs/vit-era5_${SLURM_JOB_ID}_error.txt

RDZV_PORT=${RDZV_PORT:-29500}       # default rendezvous port 29500

# No Errors dont need extra file
source /mnt/cidstore1/software/debian12/anaconda3/etc/profile.d/conda.sh
conda activate nersc24

srun torchrun \
--nnodes $SLURM_NNODES \
--nproc_per_node 1 \
--rdzv_id $RANDOM \ 
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:$RDZV_PORT \
train.py ${args}
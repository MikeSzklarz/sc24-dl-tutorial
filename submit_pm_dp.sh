#!/bin/bash

#SBATCH --job-name=vit-era5-%j         # name of the job
#SBATCH --output=logs/slurm/job_%j.txt       # output log file. MAKE SURE 'logs' DIRECTORY EXISTS OR ELSE THIS WILL ERROR 

#SBATCH --nodes=2                      # number of nodes, can be changed with -N flag
#SBATCH --ntasks-per-node=1            # number of tasks per node. We only have 1 gpu per node. Dont change this

#SBATCH --mem=65GB                    # memory per node
#SBATCH --partition=waccamaw           # partition name
#SBATCH --time=01:00:00                # time limit hrs:min:sec
#SBATCH --exclusive                    # exclusive use of node resoureces. Not sure if this works slurm 16.05.9

# Can exclude nodes with "#SBATCH --exclude=waccamaw03, waccamaw04"
# For current testing not needed.

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

NNODES=${NNODES:-$NODES}            # default to SLURM node count 1
RDZV_PORT=${RDZV_PORT:-29500}       # default rendezvous port 29500

source /mnt/cidstore1/software/debian12/anaconda3/etc/profile.d/conda.sh
conda activate nersc24

srun torchrun \
--nnodes $SLURM_NNODES \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:$RDZV_PORT \
train.py ${args}
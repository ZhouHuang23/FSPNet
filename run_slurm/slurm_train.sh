#!/bin/bash
#SBATCH --job-name=train-sod
#SBATCH --partition=multigpu

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=2
#SBATCH --output=/nfs/users/your_name/projects/cod-test-0/slurm_logs_outputs/train-model_hz_SOD_Datasets.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12341
export WORLD_SIZE=8

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
source /nfs/users/your_name/.bashrc
source /nfs/users/your_name/anaconda3/bin/activate cod

### the command to run
srun python ../train.py --path "/path_to_TrainDataset/TrainDataset" --pretrain "/path_to_Pretrian_model/preTrainModel/base_patch16_384.pth"
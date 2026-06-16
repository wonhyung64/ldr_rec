#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=JointRec
#SBATCH -o logs/s_%j.out
#SBATCH -e logs/s_%j.err
##
#SBATCH --gres=gpu:4

hostname
date
# Function to check the number of running processes
check_jobs() {
    jobs -r | wc -l
}
MAX_JOBS=4 # Maximum number of parallel jobs
experiments=(
"/home1/wonhyung64/anaconda3/envs/openmmlab/bin/python3 ./baseline/dice_lgcn.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --lr=0.001 --n-layers=4 --lambda1=0.1 --alpha=0.01  --evaluate-interval=200 --epochs=400 --data_path=/home1/wonhyung64/Github/ldr_rec/data"
"/home1/wonhyung64/anaconda3/envs/openmmlab/bin/python3 ./baseline/dice_lgcn.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --lr=0.001 --n-layers=4 --lambda1=0.1 --alpha=0.1   --evaluate-interval=200 --epochs=400 --data_path=/home1/wonhyung64/Github/ldr_rec/data"
"/home1/wonhyung64/anaconda3/envs/openmmlab/bin/python3 ./baseline/dice_lgcn.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --lr=0.001 --n-layers=4 --lambda1=0.1 --alpha=1.0   --evaluate-interval=200 --epochs=400 --data_path=/home1/wonhyung64/Github/ldr_rec/data"
"/home1/wonhyung64/anaconda3/envs/openmmlab/bin/python3 ./baseline/dice_lgcn.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --lr=0.001 --n-layers=4 --lambda1=0.3 --alpha=0.01  --evaluate-interval=200 --epochs=400 --data_path=/home1/wonhyung64/Github/ldr_rec/data"
)
for index in ${!experiments[*]}; do

    while [ "$(check_jobs)" -ge "$MAX_JOBS" ]; do
        echo "Max nodes ($MAX_JOBS) running. Waiting..."
        sleep 1m
    done

    GPU_ID=$(( COUNTER % 4 ))
    export CUDA_VISIBLE_DEVICES=$GPU_ID

    echo "Launching on GPU $GPU_ID: "
    echo ${experiments[$index]}
    ${experiments[$index]} &

    (( COUNTER++ ))
    sleep 5

done
wait

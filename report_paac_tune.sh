#!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=PAAC
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

EOF

read -r -d '' EXECUTER<<'EOF'
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

EOF



ENV=/home1/wonhyung64/anaconda3/envs/openmmlab/bin/python3
DATADIR=/home1/wonhyung64/Github/ldr_rec/data
SEED=0

# Hyperparameter search grid for PAAC
# Fixed: recdim=128, seed=0, epochs=500, evaluate-interval=500
# Tuned: lr, tau, alpha (alignment weight), gamma (contrast weight)
#
# lr    : {0.001, 0.0001}
# tau   : {0.05, 0.1, 0.2}
# alpha : {0.1, 0.5, 1.0}
# gamma : {0.1, 0.5}

experiments=(

    # ===== micro_video =====
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.05 --alpha=0.1 --gamma=0.1"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.05 --alpha=0.1 --gamma=0.5"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.05 --alpha=0.5 --gamma=0.1"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.05 --alpha=0.5 --gamma=0.5"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.05 --alpha=1.0 --gamma=0.1"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.05 --alpha=1.0 --gamma=0.5"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.1  --alpha=0.1 --gamma=0.1"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.1  --alpha=0.1 --gamma=0.5"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.1  --alpha=0.5 --gamma=0.1"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.1  --alpha=0.5 --gamma=0.5"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.1  --alpha=1.0 --gamma=0.1"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.1  --alpha=1.0 --gamma=0.5"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=0.1 --gamma=0.1"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=0.1 --gamma=0.5"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=0.5 --gamma=0.1"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=0.5 --gamma=0.5"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=1.0 --gamma=0.1"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=1.0 --gamma=0.5"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.05 --alpha=0.1 --gamma=0.1"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.05 --alpha=0.1 --gamma=0.5"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.05 --alpha=0.5 --gamma=0.1"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.05 --alpha=0.5 --gamma=0.5"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.05 --alpha=1.0 --gamma=0.1"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.05 --alpha=1.0 --gamma=0.5"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=0.1 --gamma=0.1"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=0.1 --gamma=0.5"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=0.5 --gamma=0.1"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=0.5 --gamma=0.5"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=1.0 --gamma=0.1"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=1.0 --gamma=0.5"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=0.1 --gamma=0.1"
    "./baseline/paac.py --dataset=micro_video --seed=1 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=0.1 --gamma=0.5"
    "./baseline/paac.py --dataset=micro_video --seed=2 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=0.1 --gamma=0.5"
    "./baseline/paac.py --dataset=micro_video --seed=3 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=0.1 --gamma=0.5"
    "./baseline/paac.py --dataset=micro_video --seed=4 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=0.1 --gamma=0.5"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=0.5 --gamma=0.1"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=0.5 --gamma=0.5"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=1.0 --gamma=0.1"
    # "./baseline/paac.py --dataset=micro_video --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=1.0 --gamma=0.5"

    # ===== ml-1m =====
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.05 --alpha=0.1 --gamma=0.1"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.05 --alpha=0.1 --gamma=0.5"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.05 --alpha=0.5 --gamma=0.1"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.05 --alpha=0.5 --gamma=0.5"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.05 --alpha=1.0 --gamma=0.1"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.05 --alpha=1.0 --gamma=0.5"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.1  --alpha=0.1 --gamma=0.1"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.1  --alpha=0.1 --gamma=0.5"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.1  --alpha=0.5 --gamma=0.1"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.1  --alpha=0.5 --gamma=0.5"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.1  --alpha=1.0 --gamma=0.1"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.1  --alpha=1.0 --gamma=0.5"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=0.1 --gamma=0.1"
    "./baseline/paac.py --dataset=ml-1m --seed=1 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=0.1 --gamma=0.5"
    "./baseline/paac.py --dataset=ml-1m --seed=2 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=0.1 --gamma=0.5"
    "./baseline/paac.py --dataset=ml-1m --seed=3 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=0.1 --gamma=0.5"
    "./baseline/paac.py --dataset=ml-1m --seed=4 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=0.1 --gamma=0.5"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=0.5 --gamma=0.1"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=0.5 --gamma=0.5"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=1.0 --gamma=0.1"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=1.0 --gamma=0.5"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.05 --alpha=0.1 --gamma=0.1"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.05 --alpha=0.1 --gamma=0.5"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.05 --alpha=0.5 --gamma=0.1"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.05 --alpha=0.5 --gamma=0.5"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.05 --alpha=1.0 --gamma=0.1"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.05 --alpha=1.0 --gamma=0.5"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=0.1 --gamma=0.1"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=0.1 --gamma=0.5"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=0.5 --gamma=0.1"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=0.5 --gamma=0.5"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=1.0 --gamma=0.1"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=1.0 --gamma=0.5"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=0.1 --gamma=0.1"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=0.1 --gamma=0.5"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=0.5 --gamma=0.1"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=0.5 --gamma=0.5"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=1.0 --gamma=0.1"
    # "./baseline/paac.py --dataset=ml-1m --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=1.0 --gamma=0.5"

    # ===== kuairand =====
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.05 --alpha=0.1 --gamma=0.1"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.05 --alpha=0.1 --gamma=0.5"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.05 --alpha=0.5 --gamma=0.1"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.05 --alpha=0.5 --gamma=0.5"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.05 --alpha=1.0 --gamma=0.1"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.05 --alpha=1.0 --gamma=0.5"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.1  --alpha=0.1 --gamma=0.1"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.1  --alpha=0.1 --gamma=0.5"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.1  --alpha=0.5 --gamma=0.1"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.1  --alpha=0.5 --gamma=0.5"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.1  --alpha=1.0 --gamma=0.1"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.1  --alpha=1.0 --gamma=0.5"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=0.1 --gamma=0.1"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=0.1 --gamma=0.5"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=0.5 --gamma=0.1"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=0.5 --gamma=0.5"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=1.0 --gamma=0.1"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=1.0 --gamma=0.5"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.05 --alpha=0.1 --gamma=0.1"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.05 --alpha=0.1 --gamma=0.5"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.05 --alpha=0.5 --gamma=0.1"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.05 --alpha=0.5 --gamma=0.5"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.05 --alpha=1.0 --gamma=0.1"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.05 --alpha=1.0 --gamma=0.5"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=0.1 --gamma=0.1"
    "./baseline/paac.py --dataset=kuairand --seed=1 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=0.1 --gamma=0.5"
    "./baseline/paac.py --dataset=kuairand --seed=2 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=0.1 --gamma=0.5"
    "./baseline/paac.py --dataset=kuairand --seed=3 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=0.1 --gamma=0.5"
    "./baseline/paac.py --dataset=kuairand --seed=4 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=0.1 --gamma=0.5"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=0.5 --gamma=0.1"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=0.5 --gamma=0.5"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=1.0 --gamma=0.1"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=1.0 --gamma=0.5"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=0.1 --gamma=0.1"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=0.1 --gamma=0.5"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=0.5 --gamma=0.1"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=0.5 --gamma=0.5"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=1.0 --gamma=0.1"
    # "./baseline/paac.py --dataset=kuairand --seed=$SEED --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=1.0 --gamma=0.5"

)


echo "$SLURM_SCRIPT" > runner.sh
COUNTER=0

for index in ${!experiments[*]}; do

    echo "\"$ENV ${experiments[$index]} --data_path=$DATADIR\"" >> runner.sh
    (( COUNTER++ ))

    if [ "$COUNTER" -eq 4 ]; then
        echo "$EXECUTER" >> runner.sh
        chmod +x runner.sh

        while true; do
            JOB_COUNT=$(qstat -u wonhyung64 | awk 'NR>5 {count++} END {print count}')

            if [ "$JOB_COUNT" -ge 20 ]; then
                echo "Max jobs (20) running. Waiting..."
                sleep 1m
            else
                echo "Job count is $JOB_COUNT, submitting new jobs..."
                break
            fi
        done

        sbatch runner.sh
        rm runner.sh

        echo "$SLURM_SCRIPT" >> runner.sh
        COUNTER=0

    fi

    sleep 1
done

echo "$EXECUTER" >> runner.sh
chmod +x runner.sh

while true; do
    JOB_COUNT=$(qstat -u wonhyung64 | awk 'NR>5 {count++} END {print count}')

    if [ "$JOB_COUNT" -ge 20 ]; then
        echo "Max jobs (20) running. Waiting..."
        sleep 1m
    else
        echo "Job count is $JOB_COUNT, submitting new jobs..."
        break
    fi
done

sbatch runner.sh
rm runner.sh
wait

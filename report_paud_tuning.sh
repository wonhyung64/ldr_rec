#!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=PAUDRec
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

experiments=(

    # ===== micro_video =====
    "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=1 --recdim=128 --lr=0.0001 --tau=0.05 --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=2 --recdim=128 --lr=0.0001 --tau=0.05 --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=3 --recdim=128 --lr=0.0001 --tau=0.05 --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=4 --recdim=128 --lr=0.0001 --tau=0.05 --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # # lr=0.001
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.001 --tau=0.05 --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.001 --tau=0.05 --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.001 --tau=0.05 --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.001 --tau=0.05 --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.001 --tau=0.1  --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.001 --tau=0.1  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.001 --tau=0.1  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.001 --tau=0.1  --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.001 --tau=0.5  --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.001 --tau=0.5  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.001 --tau=0.5  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.001 --tau=0.5  --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # # lr=0.0005
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.05 --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.05 --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.05 --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.05 --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.1  --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.1  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.1  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.1  --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.5  --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.5  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.5  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.5  --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # # lr=0.0001
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.05 --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.05 --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.05 --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.05 --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.5  --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.5  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.5  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=micro_video --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.5  --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"

    # ===== ml-1m =====
    "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=1 --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=2 --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=3 --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=4 --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # lr=0.001
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.001 --tau=0.05 --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.001 --tau=0.05 --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.001 --tau=0.05 --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.001 --tau=0.05 --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.001 --tau=0.1  --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.001 --tau=0.1  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.001 --tau=0.1  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.001 --tau=0.1  --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.001 --tau=0.5  --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.001 --tau=0.5  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.001 --tau=0.5  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.001 --tau=0.5  --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # # lr=0.0005
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.05 --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.05 --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.05 --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.05 --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.1  --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.1  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.1  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.1  --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.5  --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.5  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.5  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.5  --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # # lr=0.0001
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.05 --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.05 --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.05 --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.05 --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.5  --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.5  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.5  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=ml-1m --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.5  --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"

    # ===== kuairand =====
    "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=1 --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=2 --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=3 --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=4 --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # lr=0.001
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.001 --tau=0.05 --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.001 --tau=0.05 --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.001 --tau=0.05 --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.001 --tau=0.05 --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.001 --tau=0.1  --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.001 --tau=0.1  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.001 --tau=0.1  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.001 --tau=0.1  --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.001 --tau=0.5  --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.001 --tau=0.5  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.001 --tau=0.5  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.001 --tau=0.5  --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # # lr=0.0005
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.05 --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.05 --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.05 --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.05 --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.1  --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.1  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.1  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.1  --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.5  --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.5  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.5  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0005 --tau=0.5  --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # # lr=0.0001
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.05 --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.05 --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.05 --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.05 --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.5  --dropout=0.1 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.5  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.5  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "$ENV ./baseline/paud_rec.py --dataset=kuairand --seed=$SEED --recdim=128 --lr=0.0001 --tau=0.5  --dropout=0.2 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"

)


echo "$SLURM_SCRIPT" > runner.sh
COUNTER=0

for index in ${!experiments[*]}; do

    echo "\"${experiments[$index]}\"" >> runner.sh
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

        echo "$SLURM_SCRIPT" > runner.sh
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

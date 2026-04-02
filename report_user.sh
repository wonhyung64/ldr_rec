#!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
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
RANDOM_SEED=0

experiments=(

    # "expt23_non_trans.py --lr=1e-3 --recdim=64 --lambda1=0.125 --tau=0.25"
    # "expt23_non_trans.py --lr=1e-3 --recdim=64 --lambda1=0.25 --tau=0.25"
    # "expt23_non_trans.py --lr=1e-3 --recdim=64 --lambda1=0.5 --tau=0.25"
    # "expt23_non_trans.py --lr=1e-3 --recdim=64 --lambda1=0.75 --tau=0.25"
    # "expt23_non_trans.py --lr=1e-3 --recdim=64 --lambda1=0.875 --tau=0.25"
    # "expt23_non_trans.py --lr=1e-3 --recdim=128 --lambda1=0.125 --tau=0.25"
    # "expt23_non_trans.py --lr=1e-3 --recdim=128 --lambda1=0.25 --tau=0.25"
    # "expt23_non_trans.py --lr=1e-3 --recdim=128 --lambda1=0.5 --tau=0.25"
    # "expt23_non_trans.py --lr=1e-3 --recdim=128 --lambda1=0.75 --tau=0.25"
    # "expt23_non_trans.py --lr=1e-3 --recdim=128 --lambda1=0.875 --tau=0.25"
    # "expt24_simple_time.py --lr=1e-3 --recdim=64 --lambda1=0.125 --tau=0.25"
    # "expt24_simple_time.py --lr=1e-3 --recdim=64 --lambda1=0.25 --tau=0.25"
    # "expt24_simple_time.py --lr=1e-3 --recdim=64 --lambda1=0.5 --tau=0.25"
    # "expt24_simple_time.py --lr=1e-3 --recdim=64 --lambda1=0.75 --tau=0.25"
    # "expt24_simple_time.py --lr=1e-3 --recdim=64 --lambda1=0.875 --tau=0.25"
    # "expt24_simple_time.py --lr=1e-3 --recdim=128 --lambda1=0.125 --tau=0.25"
    # "expt24_simple_time.py --lr=1e-3 --recdim=128 --lambda1=0.25 --tau=0.25"
    # "expt24_simple_time.py --lr=1e-3 --recdim=128 --lambda1=0.5 --tau=0.25"
    # "expt24_simple_time.py --lr=1e-3 --recdim=128 --lambda1=0.75 --tau=0.25"
    # "expt24_simple_time.py --lr=1e-3 --recdim=128 --lambda1=0.875 --tau=0.25"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=64 --lambda1=0.125 --tau=0.25"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=64 --lambda1=0.25 --tau=0.25"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=64 --lambda1=0.5 --tau=0.25"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=64 --lambda1=0.75 --tau=0.25"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=64 --lambda1=0.875 --tau=0.25"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=128 --lambda1=0.125 --tau=0.25"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=128 --lambda1=0.25 --tau=0.25"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=128 --lambda1=0.5 --tau=0.25"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=128 --lambda1=0.75 --tau=0.25"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=128 --lambda1=0.875 --tau=0.25"

    # "expt23_non_trans.py --lr=1e-3 --recdim=64 --lambda1=0.125 --tau=0.1"
    # "expt23_non_trans.py --lr=1e-3 --recdim=64 --lambda1=0.25 --tau=0.1"
    # "expt23_non_trans.py --lr=1e-3 --recdim=64 --lambda1=0.5 --tau=0.1"
    # "expt23_non_trans.py --lr=1e-3 --recdim=64 --lambda1=0.75 --tau=0.1"
    # "expt23_non_trans.py --lr=1e-3 --recdim=64 --lambda1=0.875 --tau=0.1"
    # "expt23_non_trans.py --lr=1e-3 --recdim=128 --lambda1=0.125 --tau=0.1"
    # "expt23_non_trans.py --lr=1e-3 --recdim=128 --lambda1=0.25 --tau=0.1"
    # "expt23_non_trans.py --lr=1e-3 --recdim=128 --lambda1=0.5 --tau=0.1"
    # "expt23_non_trans.py --lr=1e-3 --recdim=128 --lambda1=0.75 --tau=0.1"
    # "expt23_non_trans.py --lr=1e-3 --recdim=128 --lambda1=0.875 --tau=0.1"
    # "expt24_simple_time.py --lr=1e-3 --recdim=64 --lambda1=0.125 --tau=0.1"
    # "expt24_simple_time.py --lr=1e-3 --recdim=64 --lambda1=0.25 --tau=0.1"
    # "expt24_simple_time.py --lr=1e-3 --recdim=64 --lambda1=0.5 --tau=0.1"
    # "expt24_simple_time.py --lr=1e-3 --recdim=64 --lambda1=0.75 --tau=0.1"
    # "expt24_simple_time.py --lr=1e-3 --recdim=64 --lambda1=0.875 --tau=0.1"
    # "expt24_simple_time.py --lr=1e-3 --recdim=128 --lambda1=0.125 --tau=0.1"
    # "expt24_simple_time.py --lr=1e-3 --recdim=128 --lambda1=0.25 --tau=0.1"
    # "expt24_simple_time.py --lr=1e-3 --recdim=128 --lambda1=0.5 --tau=0.1"
    # "expt24_simple_time.py --lr=1e-3 --recdim=128 --lambda1=0.75 --tau=0.1"
    # "expt24_simple_time.py --lr=1e-3 --recdim=128 --lambda1=0.875 --tau=0.1"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=64 --lambda1=0.125 --tau=0.1"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=64 --lambda1=0.25 --tau=0.1"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=64 --lambda1=0.5 --tau=0.1"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=64 --lambda1=0.75 --tau=0.1"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=64 --lambda1=0.875 --tau=0.1"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=128 --lambda1=0.125 --tau=0.1"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=128 --lambda1=0.25 --tau=0.1"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=128 --lambda1=0.5 --tau=0.1"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=128 --lambda1=0.75 --tau=0.1"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=128 --lambda1=0.875 --tau=0.1"

    # "expt23_non_trans.py --lr=1e-3 --recdim=64 --lambda1=0.125 --tau=0.75"
    # "expt23_non_trans.py --lr=1e-3 --recdim=64 --lambda1=0.25 --tau=0.75"
    # "expt23_non_trans.py --lr=1e-3 --recdim=64 --lambda1=0.5 --tau=0.75"
    # "expt23_non_trans.py --lr=1e-3 --recdim=64 --lambda1=0.75 --tau=0.75"
    # "expt23_non_trans.py --lr=1e-3 --recdim=64 --lambda1=0.875 --tau=0.75"
    # "expt23_non_trans.py --lr=1e-3 --recdim=128 --lambda1=0.125 --tau=0.75"
    # "expt23_non_trans.py --lr=1e-3 --recdim=128 --lambda1=0.25 --tau=0.75"
    # "expt23_non_trans.py --lr=1e-3 --recdim=128 --lambda1=0.5 --tau=0.75"
    # "expt23_non_trans.py --lr=1e-3 --recdim=128 --lambda1=0.75 --tau=0.75"
    # "expt23_non_trans.py --lr=1e-3 --recdim=128 --lambda1=0.875 --tau=0.75"
    # "expt24_simple_time.py --lr=1e-3 --recdim=64 --lambda1=0.125 --tau=0.75"
    # "expt24_simple_time.py --lr=1e-3 --recdim=64 --lambda1=0.25 --tau=0.75"
    # "expt24_simple_time.py --lr=1e-3 --recdim=64 --lambda1=0.5 --tau=0.75"
    # "expt24_simple_time.py --lr=1e-3 --recdim=64 --lambda1=0.75 --tau=0.75"
    # "expt24_simple_time.py --lr=1e-3 --recdim=64 --lambda1=0.875 --tau=0.75"
    # "expt24_simple_time.py --lr=1e-3 --recdim=128 --lambda1=0.125 --tau=0.75"
    # "expt24_simple_time.py --lr=1e-3 --recdim=128 --lambda1=0.25 --tau=0.75"
    # "expt24_simple_time.py --lr=1e-3 --recdim=128 --lambda1=0.5 --tau=0.75"
    # "expt24_simple_time.py --lr=1e-3 --recdim=128 --lambda1=0.75 --tau=0.75"
    # "expt24_simple_time.py --lr=1e-3 --recdim=128 --lambda1=0.875 --tau=0.75"

    # "expt26_debug_expt25.py --lr=1e-3 --recdim=64 --lambda1=0.125 --tau=0.75"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=64 --lambda1=0.25 --tau=0.75"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=64 --lambda1=0.5 --tau=0.75"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=64 --lambda1=0.75 --tau=0.75"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=64 --lambda1=0.875 --tau=0.75"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=128 --lambda1=0.125 --tau=0.75"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=128 --lambda1=0.25 --tau=0.75"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=128 --lambda1=0.5 --tau=0.75"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=128 --lambda1=0.75 --tau=0.75"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=128 --lambda1=0.875 --tau=0.75"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=64 --lambda1=0.125 --tau=0.5"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=64 --lambda1=0.25 --tau=0.5"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=64 --lambda1=0.5 --tau=0.5"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=64 --lambda1=0.75 --tau=0.5"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=64 --lambda1=0.875 --tau=0.5"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=128 --lambda1=0.125 --tau=0.5"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=128 --lambda1=0.25 --tau=0.5"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=128 --lambda1=0.5 --tau=0.5"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=128 --lambda1=0.75 --tau=0.5"
    # "expt26_debug_expt25.py --lr=1e-3 --recdim=128 --lambda1=0.875 --tau=0.5"

    # "expt34_time_gap_gamma.py --lr=1e-3 --recdim=128 --lambda1=0.125 --tau=0.1 --epochs=600"

    # "expt34_time_gap_gamma.py --lr=1e-2 --recdim=128 --lambda1=0.125 --tau=0.1 --evaluate-interval=599 --epochs=600 --gamma=0.05"
    # "expt34_time_gap_gamma.py --lr=5e-3 --recdim=128 --lambda1=0.125 --tau=0.1 --evaluate-interval=599 --epochs=600 --gamma=0.05"

    # "expt34_time_gap_gamma.py --lr=1e-2 --recdim=128 --lambda1=0.125 --tau=0.1 --evaluate-interval=599 --epochs=600 --gamma=0.1"
    # "expt34_time_gap_gamma.py --lr=5e-3 --recdim=128 --lambda1=0.125 --tau=0.1 --evaluate-interval=599 --epochs=600 --gamma=0.1"

    # "expt34_time_gap_gamma.py --lr=1e-3 --recdim=128 --lambda1=0.125 --tau=0.1 --evaluate-interval=20 --epochs=600 --gamma=0.05"
    # "expt34_time_gap_gamma.py --lr=1e-3 --recdim=128 --lambda1=0.125 --tau=0.1 --evaluate-interval=20 --epochs=600 --gamma=0.1"


    # "expt23_non_trans.py --lr=1e-3 --recdim=128 --lambda1=0.875 --tau=0.1 --evaluate-interval=500 --epochs=1000"
    # "expt23_non_trans.py --lr=2e-3 --recdim=128 --lambda1=0.875 --tau=0.1 --evaluate-interval=500 --epochs=1000"
    # "expt23_non_trans.py --lr=5e-3 --recdim=128 --lambda1=0.875 --tau=0.1 --evaluate-interval=500 --epochs=1000"

    # "expt32_obs_softmax_gru.py --lr=1e-3 --recdim=128 --tau=0.1 --evaluate-interval=500 --epochs=1000"
    # "expt32_obs_softmax_gru.py --lr=1e-2 --recdim=128 --tau=0.1 --evaluate-interval=500 --epochs=1000"
    # "expt32_obs_softmax_gru.py --lr=5e-3 --recdim=128 --tau=0.1 --evaluate-interval=500 --epochs=1000"
    "expt28_gru.py --lr=1e-1 --recdim=128 --tau=0.1 --evaluate-interval=500 --epochs=1000 --seed=0"
    "expt28_gru.py --lr=5e-1 --recdim=128 --tau=0.1 --evaluate-interval=500 --epochs=1000 --seed=0"
    "expt28_gru.py --lr=5e-2 --recdim=128 --tau=0.1 --evaluate-interval=500 --epochs=1000 --seed=0"

    "expt28_gru.py --lr=1e-1 --recdim=128 --tau=0.1 --evaluate-interval=500 --epochs=1000 --seed=1"
    "expt28_gru.py --lr=5e-1 --recdim=128 --tau=0.1 --evaluate-interval=500 --epochs=1000 --seed=1"
    "expt28_gru.py --lr=5e-2 --recdim=128 --tau=0.1 --evaluate-interval=500 --epochs=1000 --seed=1"

    "expt28_gru.py --lr=1e-1 --recdim=128 --tau=0.1 --evaluate-interval=500 --epochs=1000 --seed=2"
    "expt28_gru.py --lr=5e-1 --recdim=128 --tau=0.1 --evaluate-interval=500 --epochs=1000 --seed=2"
    "expt28_gru.py --lr=5e-2 --recdim=128 --tau=0.1 --evaluate-interval=500 --epochs=1000 --seed=2"

    "expt28_gru.py --lr=1e-1 --recdim=128 --tau=0.1 --evaluate-interval=500 --epochs=1000 --seed=3"
    "expt28_gru.py --lr=5e-1 --recdim=128 --tau=0.1 --evaluate-interval=500 --epochs=1000 --seed=3"
    "expt28_gru.py --lr=5e-2 --recdim=128 --tau=0.1 --evaluate-interval=500 --epochs=1000 --seed=3"

    "expt28_gru.py --lr=1e-1 --recdim=128 --tau=0.1 --evaluate-interval=500 --epochs=1000 --seed=4"
    "expt28_gru.py --lr=5e-1 --recdim=128 --tau=0.1 --evaluate-interval=500 --epochs=1000 --seed=4"
    "expt28_gru.py --lr=5e-2 --recdim=128 --tau=0.1 --evaluate-interval=500 --epochs=1000 --seed=4"

    # "expt33_obs_softmax_static.py --lr=1e-3 --recdim=128 --tau=0.1 --evaluate-interval=500 --epochs=1000"
    # "expt33_obs_softmax_static.py --lr=1e-2 --recdim=128 --tau=0.1 --evaluate-interval=500 --epochs=1000"
    # "expt33_obs_softmax_static.py --lr=5e-3 --recdim=128 --tau=0.1 --evaluate-interval=500 --epochs=1000"

    # "expt30_debug_expt29.py --lr=1e-3 --recdim=64 --lambda1=0.125 --tau=0.1"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=64 --lambda1=0.25 --tau=0.1"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=64 --lambda1=0.5 --tau=0.1"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=64 --lambda1=0.75 --tau=0.1"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=64 --lambda1=0.875 --tau=0.1"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=128 --lambda1=0.125 --tau=0.1"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=128 --lambda1=0.25 --tau=0.1"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=128 --lambda1=0.5 --tau=0.1"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=128 --lambda1=0.75 --tau=0.1"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=128 --lambda1=0.875 --tau=0.1"

    # "expt30_debug_expt29.py --lr=1e-3 --recdim=64 --lambda1=0.125 --tau=0.25"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=64 --lambda1=0.25 --tau=0.25"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=64 --lambda1=0.5 --tau=0.25"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=64 --lambda1=0.75 --tau=0.25"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=64 --lambda1=0.875 --tau=0.25"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=128 --lambda1=0.125 --tau=0.25"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=128 --lambda1=0.25 --tau=0.25"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=128 --lambda1=0.5 --tau=0.25"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=128 --lambda1=0.75 --tau=0.25"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=128 --lambda1=0.875 --tau=0.25"

    # "expt30_debug_expt29.py --lr=1e-3 --recdim=64 --lambda1=0.125 --tau=0.5"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=64 --lambda1=0.25 --tau=0.5"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=64 --lambda1=0.5 --tau=0.5"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=64 --lambda1=0.75 --tau=0.5"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=64 --lambda1=0.875 --tau=0.5"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=128 --lambda1=0.125 --tau=0.5"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=128 --lambda1=0.25 --tau=0.5"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=128 --lambda1=0.5 --tau=0.5"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=128 --lambda1=0.75 --tau=0.5"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=128 --lambda1=0.875 --tau=0.5"

    # "expt30_debug_expt29.py --lr=1e-3 --recdim=64 --lambda1=0.125 --tau=0.75"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=64 --lambda1=0.25 --tau=0.75"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=64 --lambda1=0.5 --tau=0.75"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=64 --lambda1=0.75 --tau=0.75"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=64 --lambda1=0.875 --tau=0.75"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=128 --lambda1=0.125 --tau=0.75"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=128 --lambda1=0.25 --tau=0.75"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=128 --lambda1=0.5 --tau=0.75"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=128 --lambda1=0.75 --tau=0.75"
    # "expt30_debug_expt29.py --lr=1e-3 --recdim=128 --lambda1=0.875 --tau=0.75"



    # "expt23_non_trans.py --lr=1e-3 --recdim=64 --lambda1=0.25 --seed=1"
    # "expt23_non_trans.py --lr=1e-3 --recdim=64 --lambda1=0.25 --seed=2"
    # "expt23_non_trans.py --lr=1e-3 --recdim=64 --lambda1=0.25 --seed=3"
    # "expt23_non_trans.py --lr=1e-3 --recdim=64 --lambda1=0.25 --seed=4"

    # "expt23_non_trans.py --lr=1e-3 --recdim=128 --lambda1=0.25 --seed=1"
    # "expt23_non_trans.py --lr=1e-3 --recdim=128 --lambda1=0.25 --seed=2"
    # "expt23_non_trans.py --lr=1e-3 --recdim=128 --lambda1=0.25 --seed=3"
    # "expt23_non_trans.py --lr=1e-3 --recdim=128 --lambda1=0.25 --seed=4"

    # "expt24_simple_time.py --lr=1e-3 --recdim=64 --lambda1=0.5 --seed=1"
    # "expt24_simple_time.py --lr=1e-3 --recdim=64 --lambda1=0.5 --seed=2"
    # "expt24_simple_time.py --lr=1e-3 --recdim=64 --lambda1=0.5 --seed=3"
    # "expt24_simple_time.py --lr=1e-3 --recdim=64 --lambda1=0.5 --seed=4"
    # "expt24_simple_time.py --lr=1e-3 --recdim=128 --lambda1=0.125 --seed=1"
    # "expt24_simple_time.py --lr=1e-3 --recdim=128 --lambda1=0.125 --seed=2"
    # "expt24_simple_time.py --lr=1e-3 --recdim=128 --lambda1=0.125 --seed=3"
    # "expt24_simple_time.py --lr=1e-3 --recdim=128 --lambda1=0.125 --seed=4"

)


echo "$SLURM_SCRIPT" > runner.sh
COUNTER=0

for index in ${!experiments[*]}; do

    echo "\"$ENV ${experiments[$index]}\"" >> runner.sh
    (( COUNTER++ ))

    if [ "$COUNTER" -eq 4  ]; then
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

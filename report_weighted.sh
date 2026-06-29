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
DATADIR=/home1/wonhyung64/Github/ldr_rec/data
RANDOM_SEED=0

experiments=(

    "./baseline/debiased_cf_weighted.py --model-name=mf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_cf_weighted.py --model-name=mf --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_cf_weighted.py --model-name=mf --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_cf_weighted.py --model-name=mf --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"

    "./baseline/debiased_cf_weighted.py --model-name=ncf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_cf_weighted.py --model-name=ncf --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_cf_weighted.py --model-name=ncf --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_cf_weighted.py --model-name=ncf --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"

    "./baseline/debiased_seq_rec_weighted.py --model-name=grurec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_weighted.py --model-name=grurec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_weighted.py --model-name=grurec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_weighted.py --model-name=grurec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"

    "./baseline/debiased_seq_rec_weighted.py --model-name=sasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_weighted.py --model-name=sasrec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_weighted.py --model-name=sasrec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_weighted.py --model-name=sasrec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"

    "./baseline/debiased_seq_rec_tisasrec_weighted.py --model-name=tisasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_tisasrec_weighted.py --model-name=tisasrec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_tisasrec_weighted.py --model-name=tisasrec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_tisasrec_weighted.py --model-name=tisasrec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"

    "./baseline/debiased_seq_rec_weighted.py --model-name=fearec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_weighted.py --model-name=fearec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_weighted.py --model-name=fearec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_weighted.py --model-name=fearec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"

    "./baseline/debiased_seq_rec_weighted.py --model-name=bsarec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_weighted.py --model-name=bsarec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_weighted.py --model-name=bsarec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_weighted.py --model-name=bsarec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.1"

    "./baseline/debiased_cf_weighted.py --model-name=mf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_cf_weighted.py --model-name=mf --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_cf_weighted.py --model-name=mf --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_cf_weighted.py --model-name=mf --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"

    "./baseline/debiased_cf_weighted.py --model-name=ncf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_cf_weighted.py --model-name=ncf --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_cf_weighted.py --model-name=ncf --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_cf_weighted.py --model-name=ncf --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"

    "./baseline/debiased_seq_rec_weighted.py --model-name=grurec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_weighted.py --model-name=grurec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_weighted.py --model-name=grurec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_weighted.py --model-name=grurec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"

    "./baseline/debiased_seq_rec_weighted.py --model-name=sasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_weighted.py --model-name=sasrec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_weighted.py --model-name=sasrec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_weighted.py --model-name=sasrec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"

    "./baseline/debiased_seq_rec_tisasrec_weighted.py --model-name=tisasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_tisasrec_weighted.py --model-name=tisasrec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_tisasrec_weighted.py --model-name=tisasrec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_tisasrec_weighted.py --model-name=tisasrec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"

    "./baseline/debiased_seq_rec_weighted.py --model-name=fearec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_weighted.py --model-name=fearec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_weighted.py --model-name=fearec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_weighted.py --model-name=fearec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"

    "./baseline/debiased_seq_rec_weighted.py --model-name=bsarec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_weighted.py --model-name=bsarec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_weighted.py --model-name=bsarec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_weighted.py --model-name=bsarec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.3"

    "./baseline/debiased_cf_weighted.py --model-name=mf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_cf_weighted.py --model-name=mf --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_cf_weighted.py --model-name=mf --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_cf_weighted.py --model-name=mf --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"

    "./baseline/debiased_cf_weighted.py --model-name=ncf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_cf_weighted.py --model-name=ncf --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_cf_weighted.py --model-name=ncf --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_cf_weighted.py --model-name=ncf --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"

    "./baseline/debiased_seq_rec_weighted.py --model-name=grurec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_seq_rec_weighted.py --model-name=grurec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_seq_rec_weighted.py --model-name=grurec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_seq_rec_weighted.py --model-name=grurec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"

    "./baseline/debiased_seq_rec_weighted.py --model-name=sasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_seq_rec_weighted.py --model-name=sasrec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_seq_rec_weighted.py --model-name=sasrec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_seq_rec_weighted.py --model-name=sasrec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"

    "./baseline/debiased_seq_rec_tisasrec_weighted.py --model-name=tisasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_seq_rec_tisasrec_weighted.py --model-name=tisasrec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_seq_rec_tisasrec_weighted.py --model-name=tisasrec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_seq_rec_tisasrec_weighted.py --model-name=tisasrec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"

    "./baseline/debiased_seq_rec_weighted.py --model-name=fearec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_seq_rec_weighted.py --model-name=fearec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_seq_rec_weighted.py --model-name=fearec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_seq_rec_weighted.py --model-name=fearec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"

    "./baseline/debiased_seq_rec_weighted.py --model-name=bsarec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_seq_rec_weighted.py --model-name=bsarec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_seq_rec_weighted.py --model-name=bsarec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_seq_rec_weighted.py --model-name=bsarec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.5"

    "./baseline/debiased_cf_weighted.py --model-name=mf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_cf_weighted.py --model-name=mf --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_cf_weighted.py --model-name=mf --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_cf_weighted.py --model-name=mf --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"

    "./baseline/debiased_cf_weighted.py --model-name=ncf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_cf_weighted.py --model-name=ncf --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_cf_weighted.py --model-name=ncf --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_cf_weighted.py --model-name=ncf --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"

    "./baseline/debiased_seq_rec_weighted.py --model-name=grurec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_weighted.py --model-name=grurec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_weighted.py --model-name=grurec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_weighted.py --model-name=grurec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"

    "./baseline/debiased_seq_rec_weighted.py --model-name=sasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_weighted.py --model-name=sasrec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_weighted.py --model-name=sasrec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_weighted.py --model-name=sasrec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"

    "./baseline/debiased_seq_rec_tisasrec_weighted.py --model-name=tisasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_tisasrec_weighted.py --model-name=tisasrec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_tisasrec_weighted.py --model-name=tisasrec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_tisasrec_weighted.py --model-name=tisasrec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"

    "./baseline/debiased_seq_rec_weighted.py --model-name=fearec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_weighted.py --model-name=fearec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_weighted.py --model-name=fearec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_weighted.py --model-name=fearec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"

    "./baseline/debiased_seq_rec_weighted.py --model-name=bsarec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_weighted.py --model-name=bsarec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_weighted.py --model-name=bsarec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_weighted.py --model-name=bsarec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.7"

    "./baseline/debiased_cf_weighted.py --model-name=mf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_cf_weighted.py --model-name=mf --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_cf_weighted.py --model-name=mf --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_cf_weighted.py --model-name=mf --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"

    "./baseline/debiased_cf_weighted.py --model-name=ncf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_cf_weighted.py --model-name=ncf --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_cf_weighted.py --model-name=ncf --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_cf_weighted.py --model-name=ncf --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"

    "./baseline/debiased_seq_rec_weighted.py --model-name=grurec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_weighted.py --model-name=grurec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_weighted.py --model-name=grurec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_weighted.py --model-name=grurec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"

    "./baseline/debiased_seq_rec_weighted.py --model-name=sasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_weighted.py --model-name=sasrec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_weighted.py --model-name=sasrec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_weighted.py --model-name=sasrec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"

    "./baseline/debiased_seq_rec_tisasrec_weighted.py --model-name=tisasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_tisasrec_weighted.py --model-name=tisasrec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_tisasrec_weighted.py --model-name=tisasrec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_tisasrec_weighted.py --model-name=tisasrec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"

    "./baseline/debiased_seq_rec_weighted.py --model-name=fearec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_weighted.py --model-name=fearec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_weighted.py --model-name=fearec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_weighted.py --model-name=fearec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"

    "./baseline/debiased_seq_rec_weighted.py --model-name=bsarec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_weighted.py --model-name=bsarec --dataset=micro_video --seed=2 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_weighted.py --model-name=bsarec --dataset=micro_video --seed=3 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_weighted.py --model-name=bsarec --dataset=micro_video --seed=4 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.9"
########



##############
    # "./baseline/debiased_cf_resid_only.py --model-name=mf --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_cf_resid_only.py --model-name=mf --dataset=ml-1m --seed=2 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_cf_resid_only.py --model-name=mf --dataset=ml-1m --seed=3 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_cf_resid_only.py --model-name=mf --dataset=ml-1m --seed=4 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"

    # "./baseline/debiased_cf_resid_only.py --model-name=ncf --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_cf_resid_only.py --model-name=ncf --dataset=ml-1m --seed=2 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_cf_resid_only.py --model-name=ncf --dataset=ml-1m --seed=3 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_cf_resid_only.py --model-name=ncf --dataset=ml-1m --seed=4 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"

    # "./baseline/debiased_seq_rec_resid_only.py --model-name=grurec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=grurec --dataset=ml-1m --seed=2 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=grurec --dataset=ml-1m --seed=3 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=grurec --dataset=ml-1m --seed=4 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"

    # "./baseline/debiased_seq_rec_resid_only.py --model-name=sasrec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=sasrec --dataset=ml-1m --seed=2 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=sasrec --dataset=ml-1m --seed=3 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=sasrec --dataset=ml-1m --seed=4 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"

    # "./baseline/debiased_seq_rec_tisasrec_resid_only.py --model-name=tisasrec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_tisasrec_resid_only.py --model-name=tisasrec --dataset=ml-1m --seed=2 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_tisasrec_resid_only.py --model-name=tisasrec --dataset=ml-1m --seed=3 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_tisasrec_resid_only.py --model-name=tisasrec --dataset=ml-1m --seed=4 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"

    # "./baseline/debiased_seq_rec_resid_only.py --model-name=fearec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=fearec --dataset=ml-1m --seed=2 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=fearec --dataset=ml-1m --seed=3 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=fearec --dataset=ml-1m --seed=4 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"

    # "./baseline/debiased_seq_rec_resid_only.py --model-name=bsarec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=bsarec --dataset=ml-1m --seed=2 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=bsarec --dataset=ml-1m --seed=3 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=bsarec --dataset=ml-1m --seed=4 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared"


    # "./baseline/debiased_cf_resid_only.py --model-name=mf --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_cf_resid_only.py --model-name=mf --dataset=kuairand --seed=2 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_cf_resid_only.py --model-name=mf --dataset=kuairand --seed=3 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_cf_resid_only.py --model-name=mf --dataset=kuairand --seed=4 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"

    # "./baseline/debiased_cf_resid_only.py --model-name=ncf --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_cf_resid_only.py --model-name=ncf --dataset=kuairand --seed=2 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_cf_resid_only.py --model-name=ncf --dataset=kuairand --seed=3 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_cf_resid_only.py --model-name=ncf --dataset=kuairand --seed=4 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"

    # "./baseline/debiased_seq_rec_resid_only.py --model-name=grurec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=grurec --dataset=kuairand --seed=2 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=grurec --dataset=kuairand --seed=3 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=grurec --dataset=kuairand --seed=4 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"

    # "./baseline/debiased_seq_rec_resid_only.py --model-name=sasrec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=sasrec --dataset=kuairand --seed=2 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=sasrec --dataset=kuairand --seed=3 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=sasrec --dataset=kuairand --seed=4 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"

    # "./baseline/debiased_seq_rec_tisasrec_resid_only.py --model-name=tisasrec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_tisasrec_resid_only.py --model-name=tisasrec --dataset=kuairand --seed=2 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_tisasrec_resid_only.py --model-name=tisasrec --dataset=kuairand --seed=3 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_tisasrec_resid_only.py --model-name=tisasrec --dataset=kuairand --seed=4 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"

    # "./baseline/debiased_seq_rec_resid_only.py --model-name=fearec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=fearec --dataset=kuairand --seed=2 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=fearec --dataset=kuairand --seed=3 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=fearec --dataset=kuairand --seed=4 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared"

    # "./baseline/debiased_seq_rec_resid_only.py --model-name=bsarec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.9 --c=1 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=bsarec --dataset=kuairand --seed=2 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.9 --c=1 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=bsarec --dataset=kuairand --seed=3 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.9 --c=1 --epochs=500 --ablation=shared"
    # "./baseline/debiased_seq_rec_resid_only.py --model-name=bsarec --dataset=kuairand --seed=4 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.9 --c=1 --epochs=500 --ablation=shared"

    # # ===== micro_video =====
    # # Tuning: lr {0.001, 0.005} x gamma {0.0, 0.1, 0.3, 0.5, 1.0} x dropout {0.1, 0.2}
    # # Fixed: recdim=128, seed=0, tau=0.5, depth=2, n_heads=2, epochs=1000, evaluate-interval=1000

    # "./baseline/sapid.py --dataset=micro_video --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.0 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=micro_video --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.0 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=micro_video --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.1 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=micro_video --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.1 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=micro_video --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.3 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=micro_video --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.3 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=micro_video --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.5 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=micro_video --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.5 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=micro_video --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=1.0 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=micro_video --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=1.0 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"

    # "./baseline/sapid.py --dataset=micro_video --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.0 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=micro_video --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.0 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=micro_video --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.1 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=micro_video --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.1 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=micro_video --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.3 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=micro_video --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.3 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=micro_video --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.5 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=micro_video --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.5 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=micro_video --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=1.0 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=micro_video --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=1.0 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"

    # # ===== ml-1m =====
    # "./baseline/sapid.py --dataset=ml-1m --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.0 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=ml-1m --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.0 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=ml-1m --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.1 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=ml-1m --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.1 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=ml-1m --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.3 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=ml-1m --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.3 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=ml-1m --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.5 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=ml-1m --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.5 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=ml-1m --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=1.0 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=ml-1m --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=1.0 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"

    # "./baseline/sapid.py --dataset=ml-1m --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.0 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=ml-1m --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.0 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=ml-1m --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.1 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=ml-1m --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.1 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=ml-1m --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.3 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=ml-1m --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.3 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=ml-1m --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.5 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=ml-1m --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.5 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=ml-1m --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=1.0 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=ml-1m --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=1.0 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"

    # # ===== kuairand =====
    # "./baseline/sapid.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.0 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.0 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.1 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.1 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.3 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.3 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.5 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=0.5 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=1.0 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=1.0 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"

    # "./baseline/sapid.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.0 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.0 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.1 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.1 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.3 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.3 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.5 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=0.5 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=1.0 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid.py --dataset=kuairand --seed=0 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=1.0 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"

    # # ===== micro_video =====
    # # lr=1e-4
    # "./baseline/r2rec.py --dataset=micro_video --seed=0 --lr=0.0001 --n-proto=64  --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=micro_video --seed=0 --lr=0.0001 --n-proto=128 --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=micro_video --seed=0 --lr=0.0001 --n-proto=256 --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=micro_video --seed=0 --lr=0.0001 --n-proto=64  --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=micro_video --seed=0 --lr=0.0001 --n-proto=128 --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=micro_video --seed=0 --lr=0.0001 --n-proto=256 --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # # lr=5e-4
    # "./baseline/r2rec.py --dataset=micro_video --seed=0 --lr=0.0005 --n-proto=64  --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=micro_video --seed=0 --lr=0.0005 --n-proto=128 --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=micro_video --seed=0 --lr=0.0005 --n-proto=256 --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=micro_video --seed=0 --lr=0.0005 --n-proto=64  --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=micro_video --seed=0 --lr=0.0005 --n-proto=128 --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=micro_video --seed=0 --lr=0.0005 --n-proto=256 --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # # lr=1e-3
    # "./baseline/r2rec.py --dataset=micro_video --seed=0 --lr=0.001  --n-proto=64  --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=micro_video --seed=0 --lr=0.001  --n-proto=128 --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=micro_video --seed=0 --lr=0.001  --n-proto=256 --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=micro_video --seed=0 --lr=0.001  --n-proto=64  --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=micro_video --seed=0 --lr=0.001  --n-proto=128 --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=micro_video --seed=0 --lr=0.001  --n-proto=256 --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"

    # # ===== ml-1m =====
    # # lr=1e-4
    # "./baseline/r2rec.py --dataset=ml-1m --seed=0 --lr=0.0001 --n-proto=64  --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=ml-1m --seed=0 --lr=0.0001 --n-proto=128 --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=ml-1m --seed=0 --lr=0.0001 --n-proto=256 --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=ml-1m --seed=0 --lr=0.0001 --n-proto=64  --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=ml-1m --seed=0 --lr=0.0001 --n-proto=128 --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=ml-1m --seed=0 --lr=0.0001 --n-proto=256 --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # # lr=5e-4
    # "./baseline/r2rec.py --dataset=ml-1m --seed=0 --lr=0.0005 --n-proto=64  --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=ml-1m --seed=0 --lr=0.0005 --n-proto=128 --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=ml-1m --seed=0 --lr=0.0005 --n-proto=256 --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=ml-1m --seed=0 --lr=0.0005 --n-proto=64  --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=ml-1m --seed=0 --lr=0.0005 --n-proto=128 --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=ml-1m --seed=0 --lr=0.0005 --n-proto=256 --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # # lr=1e-3
    # "./baseline/r2rec.py --dataset=ml-1m --seed=0 --lr=0.001  --n-proto=64  --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=ml-1m --seed=0 --lr=0.001  --n-proto=128 --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=ml-1m --seed=0 --lr=0.001  --n-proto=256 --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=ml-1m --seed=0 --lr=0.001  --n-proto=64  --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=ml-1m --seed=0 --lr=0.001  --n-proto=128 --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=ml-1m --seed=0 --lr=0.001  --n-proto=256 --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"

    # # ===== kuairand =====
    # # lr=1e-4
    # "./baseline/r2rec.py --dataset=kuairand --seed=0 --lr=0.0001 --n-proto=64  --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=kuairand --seed=0 --lr=0.0001 --n-proto=128 --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=kuairand --seed=0 --lr=0.0001 --n-proto=256 --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=kuairand --seed=0 --lr=0.0001 --n-proto=64  --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=kuairand --seed=0 --lr=0.0001 --n-proto=128 --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=kuairand --seed=0 --lr=0.0001 --n-proto=256 --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # # lr=5e-4
    # "./baseline/r2rec.py --dataset=kuairand --seed=0 --lr=0.0005 --n-proto=64  --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=kuairand --seed=0 --lr=0.0005 --n-proto=128 --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=kuairand --seed=0 --lr=0.0005 --n-proto=256 --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=kuairand --seed=0 --lr=0.0005 --n-proto=64  --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=kuairand --seed=0 --lr=0.0005 --n-proto=128 --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=kuairand --seed=0 --lr=0.0005 --n-proto=256 --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # # lr=1e-3
    # "./baseline/r2rec.py --dataset=kuairand --seed=0 --lr=0.001  --n-proto=64  --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=kuairand --seed=0 --lr=0.001  --n-proto=128 --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=kuairand --seed=0 --lr=0.001  --n-proto=256 --dropout=0.1 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=kuairand --seed=0 --lr=0.001  --n-proto=64  --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=kuairand --seed=0 --lr=0.001  --n-proto=128 --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"
    # "./baseline/r2rec.py --dataset=kuairand --seed=0 --lr=0.001  --n-proto=256 --dropout=0.3 --depth=2 --n-heads=2 --recdim=128 --epochs=500 --evaluate-interval=500 --pair-reset-interval=5 --contrast-size=16"

)


echo "$SLURM_SCRIPT" > runner.sh
COUNTER=0

for index in ${!experiments[*]}; do

    echo "\"$ENV ${experiments[$index]} --data_path=$DATADIR\"" >> runner.sh
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

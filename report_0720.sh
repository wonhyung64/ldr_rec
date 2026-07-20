#!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=DCRec
#SBATCH -o logs/s_%j.out
#SBATCH -e logs/s_%j.err
##
#SBATCH --gres=gpu:4

hostname
date
check_jobs() {
    jobs -r | wc -l
}
MAX_JOBS=4
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

experiments=(


#     #

#     "./baseline/debiased_cf_weighted_fixed.py --model-name=mf --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_cf_weighted_fixed.py --model-name=mf --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_cf_weighted_fixed.py --model-name=mf --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_cf_weighted_fixed.py --model-name=mf --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_cf_weighted_fixed.py --model-name=mf --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"

#     "./baseline/debiased_cf_weighted_fixed.py --model-name=ncf --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_cf_weighted_fixed.py --model-name=ncf --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_cf_weighted_fixed.py --model-name=ncf --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_cf_weighted_fixed.py --model-name=ncf --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_cf_weighted_fixed.py --model-name=ncf --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"

#     "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=grurec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=grurec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=grurec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=grurec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=grurec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"

#     "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=sasrec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=sasrec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=sasrec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=sasrec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=sasrec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"

#     "./baseline/debiased_seq_rec_tisasrec_weighted_fixed.py --model-name=tisasrec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_seq_rec_tisasrec_weighted_fixed.py --model-name=tisasrec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    # "./baseline/debiased_seq_rec_tisasrec_weighted_fixed.py --model-name=tisasrec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    # "./baseline/debiased_seq_rec_tisasrec_weighted_fixed.py --model-name=tisasrec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    # "./baseline/debiased_seq_rec_tisasrec_weighted_fixed.py --model-name=tisasrec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"

    # "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=fearec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    # "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=fearec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    # "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=fearec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    # "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=fearec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
    # "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=fearec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"

    # "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=bsarec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.9 --c=1 --epochs=500 --ablation=shared --alpha1=0.5"
    # "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=bsarec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.9 --c=1 --epochs=500 --ablation=shared --alpha1=0.5"
    # "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=bsarec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.9 --c=1 --epochs=500 --ablation=shared --alpha1=0.5"
    # "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=bsarec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.9 --c=1 --epochs=500 --ablation=shared --alpha1=0.5"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=bsarec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.9 --c=1 --epochs=500 --ablation=shared --alpha1=0.5"


    "./baseline/debiased_cf_weighted_fixed.py --model-name=mf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0."
    "./baseline/debiased_cf_weighted_fixed.py --model-name=mf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_cf_weighted_fixed.py --model-name=mf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_cf_weighted_fixed.py --model-name=mf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_cf_weighted_fixed.py --model-name=mf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_cf_weighted_fixed.py --model-name=mf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=1."

#     "./baseline/debiased_cf_weighted_fixed.py --model-name=ncf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0."
#     "./baseline/debiased_cf_weighted_fixed.py --model-name=ncf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
#     "./baseline/debiased_cf_weighted_fixed.py --model-name=ncf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
#     "./baseline/debiased_cf_weighted_fixed.py --model-name=ncf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
#     "./baseline/debiased_cf_weighted_fixed.py --model-name=ncf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
#     "./baseline/debiased_cf_weighted_fixed.py --model-name=ncf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=1."

    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=grurec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0."
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=grurec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=grurec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=grurec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=grurec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=grurec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=1."

    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=sasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0."
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=sasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=sasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=sasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=sasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=sasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=1."

    "./baseline/debiased_seq_rec_tisasrec_weighted_fixed.py --model-name=tisasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0."
    "./baseline/debiased_seq_rec_tisasrec_weighted_fixed.py --model-name=tisasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_tisasrec_weighted_fixed.py --model-name=tisasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_tisasrec_weighted_fixed.py --model-name=tisasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_tisasrec_weighted_fixed.py --model-name=tisasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_tisasrec_weighted_fixed.py --model-name=tisasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=1."

    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=fearec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0."
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=fearec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=fearec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=fearec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=fearec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=fearec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=1."

    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=bsarec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0."
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=bsarec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=bsarec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=bsarec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=bsarec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=bsarec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=1."

#     #

    "./baseline/debiased_cf_weighted_fixed.py --model-name=mf --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0."
    "./baseline/debiased_cf_weighted_fixed.py --model-name=mf --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_cf_weighted_fixed.py --model-name=mf --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_cf_weighted_fixed.py --model-name=mf --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_cf_weighted_fixed.py --model-name=mf --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_cf_weighted_fixed.py --model-name=mf --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=1."

    "./baseline/debiased_cf_weighted_fixed.py --model-name=ncf --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0."
    "./baseline/debiased_cf_weighted_fixed.py --model-name=ncf --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_cf_weighted_fixed.py --model-name=ncf --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_cf_weighted_fixed.py --model-name=ncf --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_cf_weighted_fixed.py --model-name=ncf --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_cf_weighted_fixed.py --model-name=ncf --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=1."

    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=grurec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0."
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=grurec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=grurec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=grurec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=grurec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=grurec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=1."

    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=sasrec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0."
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=sasrec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.1"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=sasrec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.3"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=sasrec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.7"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=sasrec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.9"
    "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=sasrec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=1."

#     "./baseline/debiased_seq_rec_tisasrec_weighted_fixed.py --model-name=tisasrec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_seq_rec_tisasrec_weighted_fixed.py --model-name=tisasrec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_seq_rec_tisasrec_weighted_fixed.py --model-name=tisasrec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_seq_rec_tisasrec_weighted_fixed.py --model-name=tisasrec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_seq_rec_tisasrec_weighted_fixed.py --model-name=tisasrec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"

#     "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=fearec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=fearec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=fearec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=fearec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=fearec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"

#     "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=bsarec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=bsarec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=bsarec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=bsarec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.5"
#     "./baseline/debiased_seq_rec_weighted_fixed.py --model-name=bsarec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --ablation=shared --alpha1=0.5"


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

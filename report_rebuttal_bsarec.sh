#!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=RebutBSA
#SBATCH -o logs/s_%j.out
#SBATCH -e logs/s_%j.err
##
#SBATCH --gres=gpu:2

hostname
date
check_jobs() {
    jobs -r | wc -l
}
MAX_JOBS=2
experiments=(

EOF

read -r -d '' EXECUTER<<'EOF'
)
for index in ${!experiments[*]}; do

    while [ "$(check_jobs)" -ge "$MAX_JOBS" ]; do
        echo "Max nodes ($MAX_JOBS) running. Waiting..."
        sleep 1m
    done

    GPU_ID=$(( COUNTER % 2 ))
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

# Rebuttal: BSARec x ML-1M with the same backbone configuration as the vanilla
# BSARec of Table 1 (c=5, alpha=0.9). Everything else is the submitted Ours
# setting (gamma=lambda1=0.7, tau=0.1). eta is chosen afterwards on validation by
# rebuttal_bsarec_eval.py, so --alpha1 here only affects the wandb log.
#
# Checkpoints go to ./weights_rebuttal/<config>/ml-1m/. Do NOT drop --weights_path:
# the file names do not encode c/alpha, so the default ./weights/ml-1m would resume
# from the submitted c=1 checkpoint and skip training.
#
# Phase 1: seed 1 for c=5 and c=3, plus the cosine-scored vanilla (uniform negatives,
#          same c/alpha/tau) as the same-scorer control.
# Phase 2: uncomment seeds 2-4 for the configuration that works.

OURS="--model-name=bsarec --dataset=ml-1m --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --ablation=shared --alpha1=0.5"
VANILLA="--model-name=bsarec --dataset=ml-1m --tau=0.1 --evaluate-interval=500 --epochs=500"

experiments=(

    # ---- Phase 1 ----
    "./baseline/debiased_seq_rec_weighted_fixed.py $OURS --seed=1 --c=5 --alpha=0.9 --weights_path=./weights_rebuttal/bsarec_c5_a0.9"
    "./baseline/debiased_seq_rec_weighted_fixed.py $OURS --seed=1 --c=3 --alpha=0.9 --weights_path=./weights_rebuttal/bsarec_c3_a0.9"
    "./baseline/seq_rec.py $VANILLA --seed=1 --c=5 --alpha=0.9 --weights_path=./weights_rebuttal/bsarec_vanilla_cos_c5_a0.9"

    # ---- Phase 2 (seeds 2-4) ----
    # "./baseline/debiased_seq_rec_weighted_fixed.py $OURS --seed=2 --c=5 --alpha=0.9 --weights_path=./weights_rebuttal/bsarec_c5_a0.9"
    # "./baseline/debiased_seq_rec_weighted_fixed.py $OURS --seed=3 --c=5 --alpha=0.9 --weights_path=./weights_rebuttal/bsarec_c5_a0.9"
    # "./baseline/debiased_seq_rec_weighted_fixed.py $OURS --seed=4 --c=5 --alpha=0.9 --weights_path=./weights_rebuttal/bsarec_c5_a0.9"

    # "./baseline/debiased_seq_rec_weighted_fixed.py $OURS --seed=2 --c=3 --alpha=0.9 --weights_path=./weights_rebuttal/bsarec_c3_a0.9"
    # "./baseline/debiased_seq_rec_weighted_fixed.py $OURS --seed=3 --c=3 --alpha=0.9 --weights_path=./weights_rebuttal/bsarec_c3_a0.9"
    # "./baseline/debiased_seq_rec_weighted_fixed.py $OURS --seed=4 --c=3 --alpha=0.9 --weights_path=./weights_rebuttal/bsarec_c3_a0.9"

    # "./baseline/seq_rec.py $VANILLA --seed=2 --c=5 --alpha=0.9 --weights_path=./weights_rebuttal/bsarec_vanilla_cos_c5_a0.9"
    # "./baseline/seq_rec.py $VANILLA --seed=3 --c=5 --alpha=0.9 --weights_path=./weights_rebuttal/bsarec_vanilla_cos_c5_a0.9"
    # "./baseline/seq_rec.py $VANILLA --seed=4 --c=5 --alpha=0.9 --weights_path=./weights_rebuttal/bsarec_vanilla_cos_c5_a0.9"
)



echo "$SLURM_SCRIPT" > runner.sh
COUNTER=0

for index in ${!experiments[*]}; do

    echo "\"$ENV ${experiments[$index]} --data_path=$DATADIR\"" >> runner.sh
    (( COUNTER++ ))

    if [ "$COUNTER" -eq 2  ]; then
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

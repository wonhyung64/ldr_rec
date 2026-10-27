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
# setting (gamma=lambda1=0.7, tau=0.1, dr-anchor=user, ablation=shared). eta is
# chosen afterwards on validation by rebuttal_bsarec_eval.py.
#
# Uses rebuttal-only copies of the training scripts (same training loop, no wandb,
# no in-training eval). They always save to ./weights_rebuttal/ml-1m/ with names
#   rebuttal_ours_bsarec_c{c}_a{alpha}_lambda{lambda1}_e{epoch}_seed{seed}.pt
#   rebuttal_vanilla_cos_bsarec_c{c}_a{alpha}_e{epoch}_seed{seed}.pt
# so existing checkpoints in ./weights are never loaded or overwritten.
#
# Phase 1: seed 1 for c=5 and c=3, plus the cosine-scored vanilla (uniform negatives,
#          same c/alpha/tau) as the same-scorer control.
# Phase 2: uncomment seeds 2-4 for the configuration that works.

OURS="--dataset=ml-1m --tau=0.1 --lambda1=0.7 --pair-reset-interval=5 --epochs=500"
VANILLA="--dataset=ml-1m --tau=0.1 --epochs=500"

experiments=(

    # ---- Phase 1 ----
    "./baseline/rebuttal_bsarec_ours.py $OURS --seed=1 --c=5 --alpha=0.9"
    "./baseline/rebuttal_bsarec_ours.py $OURS --seed=1 --c=3 --alpha=0.9"
    "./baseline/rebuttal_bsarec_vanilla.py $VANILLA --seed=1 --c=5 --alpha=0.9"

    # ---- Phase 2 (seeds 2-4) ----
    # "./baseline/rebuttal_bsarec_ours.py $OURS --seed=2 --c=5 --alpha=0.9"
    # "./baseline/rebuttal_bsarec_ours.py $OURS --seed=3 --c=5 --alpha=0.9"
    # "./baseline/rebuttal_bsarec_ours.py $OURS --seed=4 --c=5 --alpha=0.9"

    # "./baseline/rebuttal_bsarec_ours.py $OURS --seed=2 --c=3 --alpha=0.9"
    # "./baseline/rebuttal_bsarec_ours.py $OURS --seed=3 --c=3 --alpha=0.9"
    # "./baseline/rebuttal_bsarec_ours.py $OURS --seed=4 --c=3 --alpha=0.9"

    # "./baseline/rebuttal_bsarec_vanilla.py $VANILLA --seed=2 --c=5 --alpha=0.9"
    # "./baseline/rebuttal_bsarec_vanilla.py $VANILLA --seed=3 --c=5 --alpha=0.9"
    # "./baseline/rebuttal_bsarec_vanilla.py $VANILLA --seed=4 --c=5 --alpha=0.9"
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

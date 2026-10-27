#!/bin/bash
# AAAI-27 rebuttal (VShq): static popularity with exponent alpha, pi_a(v) ∝ n_v^a  (= 1/pop^a correction)
#   alpha=1 is the Table 3 static row, alpha=0 is uniform negatives (vanilla backbone, Table 1).
#   Every run evaluates eta in {0.1,0.3,0.5,0.7,0.9} from one checkpoint; eta=0.5 is the Table 3 protocol.
#   Results: results/static_pop/{dataset}_{model}_pp{alpha}_seed{seed}.json -> python rebuttal_static_alpha_table.py
#
# usage (run on the SLURM server; paths are relative to the repo root):
#   bash report_rebuttal_static_alpha.sh status   # list runs: done / pending / no checkpoint
#   bash report_rebuttal_static_alpha.sh dry      # print what each sbatch job would run, submit nothing
#   bash report_rebuttal_static_alpha.sh          # submit pending runs (finished runs are skipped)
#
#   SEEDS="2 3 4" ALPHAS="0.5" bash report_rebuttal_static_alpha.sh   # e.g. extra seeds later

MODE=${1:-submit}
SEEDS=${SEEDS:-1}
ALPHAS=${ALPHAS:-"0.5 1.5"}

cd "$(dirname "$0")" || exit 1
mkdir -p logs results/static_pop

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=StaticAlpha
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

RUNS_PER_JOB=2   # keep equal to --gres=gpu:N, MAX_JOBS and COUNTER % N above


ENV=/home1/wonhyung64/anaconda3/envs/openmmlab/bin/python3
DATADIR=/home1/wonhyung64/Github/ldr_rec/data
COMMON="--tau=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"

# model  dataset  script  lambda1 (only names checkpoints; same values as the Table 3 static runs)
CELLS=(
    "sasrec micro_video ./baseline/debiased_seq_rec_static_prior.py 0.9"
    "mf     micro_video ./baseline/debiased_cf_static_prior.py      0.1"
    "sasrec ml-1m       ./baseline/debiased_seq_rec_static_prior.py 0.3"
    "mf     ml-1m       ./baseline/debiased_cf_static_prior.py      0.5"
    "sasrec kuairand    ./baseline/debiased_seq_rec_static_prior.py 0.3"
    "mf     kuairand    ./baseline/debiased_cf_static_prior.py      0.5"
)

experiments=()
N_DONE=0
N_SKIP=0

add_run() {  # model dataset script lambda1 alpha seed [--eval-only]
    local model=$1 dataset=$2 script=$3 lambda1=$4 alpha=$5 seed=$6 extra=$7
    local json=results/static_pop/${dataset}_${model}_pp${alpha}_seed${seed}.json
    local tag="${model}/${dataset} alpha=${alpha} seed=${seed}${extra:+ (eval-only)}"

    if [ -f "$json" ]; then
        (( N_DONE++ ))
        [ "$MODE" = "status" ] && echo "  [done]    $tag"
        return
    fi
    if [ -n "$extra" ] && [ ! -f "weights/${dataset}/${model}_lambda${lambda1}_e500_seed${seed}_static_prior.pt" ]; then
        (( N_SKIP++ ))
        [ "$MODE" = "status" ] && echo "  [no ckpt] $tag -> compare with the Table 3 static value instead"
        return
    fi
    [ "$MODE" = "status" ] && echo "  [pending] $tag"
    experiments+=("$script --model-name=$model --dataset=$dataset --lambda1=$lambda1 --pop-power=$alpha --seed=$seed $extra $COMMON")
}

# training runs: alpha != 1 (two alphas of the same cell share a job, so job lengths match)
[ "$MODE" = "status" ] && echo "training runs:"
for seed in $SEEDS; do
    for cell in "${CELLS[@]}"; do
        read -r model dataset script lambda1 <<< "$cell"
        for alpha in $ALPHAS; do
            add_run "$model" "$dataset" "$script" "$lambda1" "$alpha" "$seed"
        done
    done
done

# eval-only: re-score the existing Table 3 alpha=1 checkpoints on the eta grid (no training)
[ "$MODE" = "status" ] && echo "alpha=1 eval-only runs:"
for seed in $SEEDS; do
    for cell in "${CELLS[@]}"; do
        read -r model dataset script lambda1 <<< "$cell"
        add_run "$model" "$dataset" "$script" "$lambda1" "1.0" "$seed" "--eval-only"
    done
done

echo "pending: ${#experiments[@]}, done: $N_DONE, alpha=1 without checkpoint: $N_SKIP"
[ "$MODE" = "status" ] && exit 0
[ ${#experiments[@]} -eq 0 ] && { echo "Nothing to submit."; exit 0; }


JOB_IDX=0
submit_runner() {
    echo "$EXECUTER" >> runner.sh
    chmod +x runner.sh
    (( JOB_IDX++ ))

    if [ "$MODE" = "dry" ]; then
        echo "--- job $JOB_IDX (not submitted)"
        grep "^\"$ENV" runner.sh
        rm runner.sh
        return
    fi

    while true; do
        JOB_COUNT=$(qstat -u wonhyung64 | awk 'NR>5 {count++} END {print count}')

        if [ "${JOB_COUNT:-0}" -ge 20 ]; then
            echo "Max jobs (20) running. Waiting..."
            sleep 1m
        else
            echo "Job count is ${JOB_COUNT:-0}, submitting new jobs..."
            break
        fi
    done

    sbatch runner.sh
    rm runner.sh
}


echo "$SLURM_SCRIPT" > runner.sh
COUNTER=0

for index in ${!experiments[*]}; do

    echo "\"$ENV ${experiments[$index]} --data_path=$DATADIR\"" >> runner.sh
    (( COUNTER++ ))

    if [ "$COUNTER" -eq "$RUNS_PER_JOB" ]; then
        submit_runner
        echo "$SLURM_SCRIPT" > runner.sh
        COUNTER=0
    fi

    [ "$MODE" = "dry" ] || sleep 1
done

if [ "$COUNTER" -gt 0 ]; then
    submit_runner
else
    rm -f runner.sh
fi
wait

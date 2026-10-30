#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=RebutEta0Tail
#SBATCH -o logs/s_%A_%a.out
#SBATCH -e logs/s_%A_%a.err
##
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

# AAAI-27 rebuttal (CW75: gain of the debiased score alone): eval-only re-scoring of the Table 1
# checkpoints (Ours timefix + vanilla _backbone_, 6 models, seeds 1-4) on the eta grid, reporting
# head/tail Recall@10 (Figure 3 splits), tail share of top-10 and Spearman(score, popularity).
# One array task per dataset:
#   sbatch report_rebuttal_eta0_tail.sh
# then, once all three tasks are done (login node is fine, no GPU needed):
#   python ./rebuttal_eta0_tail.py --summarize | tee results/rebuttal_eta0_tail/summary.md
# Results are cached per checkpoint (results/rebuttal_eta0_tail/<ds>_<model>_<kind>_seed<k>.json);
# missing checkpoints are listed as [missing] in the log and skipped.

ENV=/home1/wonhyung64/anaconda3/envs/openmmlab/bin/python3
DATASETS=(micro_video ml-1m kuairand)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}
mkdir -p logs results/rebuttal_eta0_tail

hostname
date
echo "dataset=$DS"

$ENV ./rebuttal_eta0_tail.py --dataset=$DS

date

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=RebutEtaStd
#SBATCH -o logs/s_%j.out
#SBATCH -e logs/s_%j.err
##
#SBATCH --gres=gpu:1

# AAAI-27 rebuttal (VShq error bars / CW75 eta=0): eval-only re-scoring of the Table 1 checkpoints
# (Ours timefix + vanilla _backbone_, seeds 1-4) on the eta grid -> std of test Recall@10 at eta=0 vs eta.
#   sbatch report_rebuttal_eta_std.sh
# Results are cached per checkpoint (results/rebuttal_eta_std/<ds>_<model>_<kind>_seed<k>.json);
# missing checkpoints are listed as [missing] in the log and skipped.

ENV=/home1/wonhyung64/anaconda3/envs/openmmlab/bin/python3
mkdir -p logs results/rebuttal_eta_std

hostname
date

for DS in micro_video ml-1m kuairand; do
    $ENV ./rebuttal_eta_std.py --dataset=$DS
done
$ENV ./rebuttal_eta_std.py --summarize | tee results/rebuttal_eta_std/summary.md
$ENV ./rebuttal_eta_std.py --summarize --sample-std | tee results/rebuttal_eta_std/summary_sample_std.md

date

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=RebutEval
#SBATCH -o logs/s_%j.out
#SBATCH -e logs/s_%j.err
##
#SBATCH --gres=gpu:1

# Run after the report_rebuttal_bsarec.sh jobs have finished:  sbatch report_rebuttal_bsarec_eval.sh
# Results are cached per checkpoint (results/rebuttal_bsarec/ml-1m/<label>_seed<k>.json), so
# re-submitting after Phase 2 only evaluates the new seeds.

ENV=/home1/wonhyung64/anaconda3/envs/openmmlab/bin/python3

hostname
date

$ENV ./rebuttal_bsarec_eval.py --dataset=ml-1m --with-reference
$ENV ./rebuttal_bsarec_eval.py --dataset=ml-1m --summarize | tee results/rebuttal_bsarec/ml-1m_summary.md

date

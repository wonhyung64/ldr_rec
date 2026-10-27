# ldr_rec

Popularity-debiased recommendation experiments (MF / SASRec / TiSASRec backbones) with debiasing baselines such as DICE, DCRec, DDC, MELT, PAAC and PAUD.

## Structure

- `baseline/` – model training scripts (`*.py`) and shared modules (`module/`)
- `preprocess_*.py` – dataset preprocessing (ML-1M, ML-100K, Beauty, Fashion, Clothing)
- `data/` – datasets (micro_video, kuairand, ml-1m, …)
- `collapse_analysis/` – embedding-collapse analysis and paper figures/tables
- `report_*.sh` – SLURM experiment launchers
- `plot_*.py` – sensitivity and head/tail plots

## Usage

```bash
# 1. preprocess a dataset
python preprocess_ml1m.py

# 2. train (see baseline/module/utils.py for all arguments)
cd baseline
python debiased_seq_rec_weighted.py --dataset micro_video --model-name sasrec --seed 0
```

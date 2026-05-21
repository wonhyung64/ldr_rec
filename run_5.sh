#!/bin/bash


experiments=(

    # "./baseline/debiased_seq_rec.py --model-name=sasrec --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    # "./baseline/debiased_seq_rec.py --model-name=sasrec --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    # "./baseline/debiased_seq_rec.py --model-name=sasrec --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    # "./baseline/debiased_seq_rec.py --model-name=bsarec --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --alpha=0.9 --c=1"

    # "./baseline/debiased_seq_rec.py --model-name=bsarec --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --alpha=0.9 --c=1"
    # "./baseline/debiased_seq_rec.py --model-name=bsarec --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --alpha=0.9 --c=1"
    # "./baseline/debiased_seq_rec.py --model-name=bsarec --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --alpha=0.9 --c=1"
    # "./baseline/debiased_seq_rec.py --model-name=fearec --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"

    ##################3


    # "./baseline/debiased_seq_rec.py --model-name=bsarec --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --alpha=0.9 --c=1"
    # "./baseline/debiased_seq_rec.py --model-name=fearec --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    # "./baseline/debiased_seq_rec.py --model-name=fearec --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    # "./baseline/debiased_seq_rec.py --model-name=fearec --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    # "./baseline/debiased_seq_rec.py --model-name=fearec --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"

    #############3

    # "./baseline/debiased_seq_rec.py --model-name=grurec --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    # "./baseline/debiased_cf.py --model-name=mf --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    # "./baseline/debiased_cf.py --model-name=mf --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    # "./baseline/debiased_cf.py --model-name=mf --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"

    # "./baseline/debiased_cf.py --model-name=mf --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    # "./baseline/debiased_cf.py --model-name=mf --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    # "./baseline/debiased_cf.py --model-name=ncf --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    # "./baseline/debiased_cf.py --model-name=ncf --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"

    "./baseline/debiased_cf.py --model-name=ncf --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    "./baseline/debiased_cf.py --model-name=ncf --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    "./baseline/debiased_cf.py --model-name=ncf --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.9 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"

)

ENV=/home1/wonhyung64/anaconda3/envs/openmmlab/bin/python3
DATADIR=/home1/wonhyung64/Github/ldr_rec/data
DEVICE0=cuda:0
DEVICE1=cuda:1
DEVICE2=cuda:2
DEVICE3=cuda:3

$ENV ${experiments[0]} --data_path=$DATADIR --device=$DEVICE0 &
sleep 20
$ENV ${experiments[1]} --data_path=$DATADIR --device=$DEVICE1 &
sleep 20
$ENV ${experiments[2]} --data_path=$DATADIR --device=$DEVICE2 & 
# sleep 20
# $ENV ${experiments[3]} --data_path=$DATADIR --device=$DEVICE3 & 

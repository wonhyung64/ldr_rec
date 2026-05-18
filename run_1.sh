#!/bin/bash


experiments=(

    "./baseline/debiased_seq_rec.py --model-name=sasrec --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.1 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    "./baseline/debiased_seq_rec.py --model-name=sasrec --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.3 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    "./baseline/debiased_seq_rec.py --model-name=sasrec --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.5 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    "./baseline/debiased_seq_rec.py --model-name=sasrec --dataset=kuairand --seed=0 --tau=0.1 --lambda1=0.7 --dr-anchor=user --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"

)

ENV=/home1/wonhyung64/anaconda3/envs/openmmlab/bin/python3
DATADIR=/home1/wonhyung64/Github/ldr_rec/data
DEVICE0=cuda:0
DEVICE1=cuda:1
DEVICE2=cuda:2
DEVICE3=cuda:3

$ENV ${experiments[0]} --data_path=$DATADIR --device=$DEVICE0 &
$ENV ${experiments[1]} --data_path=$DATADIR --device=$DEVICE1 &
$ENV ${experiments[2]} --data_path=$DATADIR --device=$DEVICE2 & 
$ENV ${experiments[3]} --data_path=$DATADIR --device=$DEVICE3 & 

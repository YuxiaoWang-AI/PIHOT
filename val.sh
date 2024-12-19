#!/usr/bin/env bash

EPOCH_NUM=12
for epoch in `seq 12 $EPOCH_NUM`
do
    CUDA_VISIBLE_DEVICES=3 python eval_metric_epoch_val.py --cfg config/hot-resnet50dilated-c1.yaml --epoch ${epoch}
done
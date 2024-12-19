#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3 python eval_metric_epoch_test.py --cfg config/hot-resnet50dilated-c1.yaml --epoch 12

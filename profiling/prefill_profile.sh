#!/bin/bash

for i in 1 2 4
do
    VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=3 python prefill_profile.py --in-token-size 8192 --batch-size $i
    sleep 5
done
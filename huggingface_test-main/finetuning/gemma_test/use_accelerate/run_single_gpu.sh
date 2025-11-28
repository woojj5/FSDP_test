#!/bin/bash
# 단일 GPU 실행 스크립트

cd /data2/jeOn9/huggingface_test-main/huggingface_test-main/finetuning/gemma_test/use_accelerate

# GPU 0번 사용 (다른 GPU 사용하려면 숫자 변경)
CUDA_VISIBLE_DEVICES=0 python fsdp_train.py


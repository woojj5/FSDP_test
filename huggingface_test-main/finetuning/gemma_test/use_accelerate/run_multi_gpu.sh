#!/bin/bash
# FSDP 멀티 GPU 실행 스크립트

cd /data2/jeOn9/huggingface_test-main/huggingface_test-main/finetuning/gemma_test/use_accelerate

# accelerate_config.yaml을 사용하여 FSDP 실행
accelerate launch --config_file accelerate_config.yaml fsdp_train.py


# fsdp_train.py GPU 실행 가이드

## 현재 상태
- 파일명은 `fsdp_train.py`이지만 실제로는 FSDP를 사용하지 않고 일반 Transformers Trainer를 사용합니다
- `accelerate_config.yaml`은 FSDP 8 GPU 설정이지만, 현재 코드와는 연결되지 않습니다

## 실행 방법

### 방법 1: 단일 GPU 실행 (가장 간단)

```bash
cd /data2/jeOn9/huggingface_test-main/huggingface_test-main/finetuning/gemma_test/use_accelerate

# 특정 GPU 지정 (예: GPU 0번 사용)
CUDA_VISIBLE_DEVICES=0 python fsdp_train.py

# 또는 자동으로 사용 가능한 GPU 사용
python fsdp_train.py
```

### 방법 2: 멀티 GPU 실행 (DataParallel 방식)

```bash
cd /data2/jeOn9/huggingface_test-main/huggingface_test-main/finetuning/gemma_test/use_accelerate

# accelerate를 사용한 멀티 GPU (FSDP 없이)
accelerate launch --num_processes 2 --mixed_precision bf16 fsdp_train.py

# 또는 config 파일 사용 (하지만 FSDP는 적용 안됨)
accelerate launch --config_file accelerate_config.yaml fsdp_train.py
```

### 방법 3: torchrun을 사용한 멀티 GPU

```bash
cd /data2/jeOn9/huggingface_test-main/huggingface_test-main/finetuning/gemma_test/use_accelerate

# 2개 GPU 사용 예시
torchrun --nproc_per_node=2 fsdp_train.py

# 4개 GPU 사용 예시
torchrun --nproc_per_node=4 fsdp_train.py

# 8개 GPU 사용 예시
torchrun --nproc_per_node=8 fsdp_train.py
```

## GPU 확인

실행 전에 GPU가 제대로 인식되는지 확인:

```bash
# GPU 확인
nvidia-smi

# PyTorch에서 GPU 인식 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## 주의사항

1. **현재 코드는 FSDP를 사용하지 않습니다**
   - 파일명과 달리 실제로는 일반 Trainer 사용
   - FSDP를 사용하려면 코드 수정 필요

2. **TrainingArguments에 FSDP 설정 추가 필요**
   - FSDP를 사용하려면 `fsdp` 파라미터 추가 필요
   - 예: `fsdp=["full_shard", "auto_wrap"]`

3. **메모리 부족 시**
   - `per_device_train_batch_size` 줄이기
   - `gradient_accumulation_steps` 늘리기
   - `gradient_checkpointing=True` 유지 (이미 설정됨)

## 빠른 실행 스크립트

아래 스크립트를 사용하면 편리합니다:

```bash
# run_single_gpu.sh
#!/bin/bash
cd /data2/jeOn9/huggingface_test-main/huggingface_test-main/finetuning/gemma_test/use_accelerate
CUDA_VISIBLE_DEVICES=0 python fsdp_train.py
```

```bash
# run_multi_gpu.sh (예: 4개 GPU)
#!/bin/bash
cd /data2/jeOn9/huggingface_test-main/huggingface_test-main/finetuning/gemma_test/use_accelerate
torchrun --nproc_per_node=4 fsdp_train.py
```


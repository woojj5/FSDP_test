# Gemma-2B FSDP 파인튜닝 가이드

`google/gemma-2b-it` 모델을 FSDP(Fully Sharded Data Parallel) 기술과 LoRA를 사용하여 멀티 GPU 환경에서 파인튜닝하는 코드입니다.

## 환경 요구사항

- **GPU**: NVIDIA GeForce RTX 4090 4개 (현재 설정 기준, `accelerate_config.yaml`의 `num_processes: 4`)
- **라이브러리 버전**:
  - `transformers==4.56.2`
  - `accelerate==1.10.1`
  - `peft==0.17.1`
  - `trl==0.23.0`
  - `torch>=2.0.0`
  - `datasets`
  - `bitsandbytes`


## 주요 특징

### 1. FSDP 최적화
- **FULL_SHARD**: 모델 파라미터를 모든 GPU에 분산
- **TRANSFORMER_BASED_WRAP**: 트랜스포머 레이어별로 자동 래핑
- **Activation Checkpointing**: FSDP 전용 메모리 최적화 (gradient checkpointing 대신 사용)
- **GemmaDecoderLayer 래핑**: Gemma 모델 구조에 최적화된 레이어 래핑

### 2. LoRA 설정
- **Rank**: 8
- **Alpha**: 16
- **Dropout**: 0.05
- **Target Modules**: `q_proj`, `o_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`, `down_proj`
- **학습 가능 파라미터**: 약 980만 개 (전체의 0.39%)

### 3. 학습 설정
- **Mixed Precision**: bf16
- **Gradient Accumulation**: 8 steps
- **Batch Size**: per_device_train_batch_size=1 (effective batch size = 8)


## 실행 방법

### 1. 환경 설정
```bash
# 필요한 라이브러리 설치
pip install transformers==4.56.2 accelerate==1.10.1 peft==0.17.1 trl==0.23.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install datasets bitsandbytes
```

### 2. Accelerate 설정 확인 및 수정

현재 `accelerate_config.yaml`은 **4개 GPU**를 기준으로 설정되어 있습니다:

```yaml
num_processes: 4  # 현재 4개 GPU 설정
```

GPU 개수가 다르면 `num_processes`를 실제 GPU 개수에 맞게 수정하세요.

```bash
cd finetuning/gemma_test/use_accelerate
# GPU 개수 확인
nvidia-smi --list-gpus | wc -l

# 설정 파일 수정 (필요시)
# accelerate config --config_file accelerate_config.yaml
```

### 3. 파인튜닝 실행

```bash
cd finetuning/gemma_test/use_accelerate

# FSDP 실행 (accelerate_config.yaml 사용)
accelerate launch --config_file accelerate_config.yaml fsdp_train.py

# 또는 실행 스크립트 사용
./run_multi_gpu.sh
```

## 주요 코드 구조

### FSDP 설정
```python
# fsdp_config를 사용하여 FSDP 전용 설정
fsdp_config = {
    "transformer_layer_cls_to_wrap": ["GemmaDecoderLayer"],
    "activation_checkpointing": True,  # gradient checkpointing 대신 사용
}

training_args = TrainingArguments(
    fsdp="full_shard auto_wrap",
    fsdp_config=fsdp_config,
    # ...
)
```

### 주의사항
- **FSDP 사용 시**: `gradient_checkpointing` 대신 `fsdp_config`의 `activation_checkpointing` 사용
- **GPU 개수**: `accelerate_config.yaml`의 `num_processes`를 실제 GPU 개수에 맞게 설정
- **Deprecation 해결**: `dtype` 사용 (torch_dtype 대신), `trainer.processing_class` 사용 (tokenizer 대신)

## 결과 확인
학습 완료 후:
- **체크포인트**: `./outputs/` 디렉토리
- **학습 가능 파라미터**: 약 980만 개 (전체의 0.39%)
- **LoRA 어뎁터**: 체크포인트에 저장됨

## 트러블슈팅

### "Duplicate GPU detected" 오류
- 원인: `num_processes`가 실제 GPU 개수보다 많음
- 해결: `accelerate_config.yaml`의 `num_processes`를 실제 GPU 개수로 수정

### 메모리 부족
- `per_device_train_batch_size` 줄이기
- `gradient_accumulation_steps` 늘리기
- `activation_checkpointing`이 이미 활성화되어 있음

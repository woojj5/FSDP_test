import torch
import os
import glob
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

dataset = load_dataset("daekeun-ml/naver-news-summarization-ko")

BASE_MODEL = "google/gemma-2b-it"

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token

def formatting_func(example):
    text = f"""<bos><start_of_turn>user
다음 글을 요약해주세요:

{example['document']}<end_of_turn>
<start_of_turn>model
{example['summary']}<end_of_turn><eos>"""
    return text

# LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,  # torch_dtype -> dtype (deprecation 해결)
    low_cpu_mem_usage=True,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# FSDP 설정 (accelerate_config.yaml과 함께 사용)
# FSDP를 사용할 때는 gradient_checkpointing 대신 fsdp_config의 activation_checkpointing 사용
fsdp_config = {
    "transformer_layer_cls_to_wrap": ["GemmaDecoderLayer"],  # Gemma 모델 레이어
    "activation_checkpointing": True,  # gradient checkpointing 대신 사용
}

# 체크포인트 자동 감지 및 재개 설정
output_dir = "outputs"
resume_from_checkpoint = None

# 환경 변수에서 체크포인트 경로 확인
if os.environ.get("RESUME_CHECKPOINT"):
    resume_from_checkpoint = os.environ.get("RESUME_CHECKPOINT")
    print(f"[체크포인트 재개] 환경 변수에서 지정된 체크포인트: {resume_from_checkpoint}")
elif os.path.exists(output_dir):
    # 가장 최근 체크포인트 자동 찾기
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if checkpoints:
        # checkpoint 번호로 정렬하여 가장 최근 것 선택
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        resume_from_checkpoint = latest_checkpoint
        print(f"[체크포인트 재개] 자동 감지된 최신 체크포인트: {resume_from_checkpoint}")
        print(f"  → 이전 학습을 이어서 진행합니다 (더 빠릅니다!)")
    else:
        print(f"[새 학습 시작] 체크포인트를 찾을 수 없습니다. 처음부터 학습합니다.")

# 학습 설정
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_steps=50,
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    push_to_hub=False,
    report_to='none',
    # FSDP 설정 (accelerate_config.yaml과 함께 사용)
    fsdp="full_shard auto_wrap",  # FSDP 활성화
    fsdp_config=fsdp_config,  # FSDP 상세 설정
    resume_from_checkpoint=resume_from_checkpoint,  # 체크포인트에서 재개
)

# SFTTrainer - FSDP 사용
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    formatting_func=formatting_func,
)

# 토크나이저 수동 설정 (deprecation warning 해결)
trainer.processing_class = tokenizer

# 학습 시작
if resume_from_checkpoint:
    print(f"\n{'='*60}")
    print(f"체크포인트에서 재개: {resume_from_checkpoint}")
    print(f"이전 학습을 이어서 진행합니다 (처음부터 시작하는 것보다 빠릅니다)")
    print(f"{'='*60}\n")
else:
    print(f"\n{'='*60}")
    print(f"새로운 학습 시작")
    print(f"{'='*60}\n")

trainer.train()

# yaml에서 설정
# from accelerate import Accelerator
# # 학습 완료 후 저장
# accelerator = Accelerator()
# accelerator.wait_for_everyone()

# if accelerator.is_main_process:
#     print("Saving LoRA adapter...")
#     unwrapped_model = accelerator.unwrap_model(trainer.model)
#     unwrapped_model.save_pretrained("lora_adapter_final")
#     tokenizer.save_pretrained("lora_adapter_final")
#     print("Done!")
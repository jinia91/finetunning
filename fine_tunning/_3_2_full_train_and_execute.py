import torch
from datasets import Dataset
from sympy.core.evalf import evalf_prod
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, TaskType
from trl import SFTTrainer
import numpy as np

# 모델 로드
model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval() # 평가 모드로 설정


# 토크나이저 로드 및 설정
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)


# 학습 데이터 정의
data = {
    "text": [
        "질문: 우리집 강아지 이름은?\n 정답: 순둥이",
        "질문: 바다는 왜 파란가요?\n 정답: 햇빛의 산란",
    ]
}
dataset = Dataset.from_dict(data)

# Decide optimizer & precision for NVIDIA CUDA + bitsandbytes
use_bnb = False
optim_name = "paged_adamw_8bit" if use_bnb else "adamw_torch"
bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
fp16_ok = torch.cuda.is_available() and not bf16_ok

training_params = TrainingArguments(
    num_train_epochs=1,
    max_steps=30,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim=optim_name,
    warmup_ratio=0.03,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    bf16=bf16_ok,
    fp16=fp16_ok,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    logging_steps=1,
    save_steps=100,
    save_total_limit=2,
    push_to_hub=False,
    report_to="none",
)

# SFTTrainer 설정
trainer = SFTTrainer(
    model=model,
    peft_config=lora_config,
    train_dataset=dataset,
    eval_dataset=dataset,
    args=training_params,
)

# 학습 시작!
trainer.train()


# 학습 완료 후 테스트
peft_model = trainer.model
peft_model.eval()

# 2) Llama-3 채팅 템플릿 사용
messages = [
    {"role": "system", "content": "한국어로 짧고 정확하게만 답하세요. 모르면 모른다고 해라"},
    {"role": "user", "content": "우리집 강아지 이름은?"}
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(peft_model.device)

# 3) 종료 토큰 지정 (Llama-3는 <|eot_id|>도 있음)
eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>") if "<|eot_id|>" in tokenizer.get_vocab() else None
eos = [tokenizer.eos_token_id] + ([eot_id] if eot_id is not None else [])

outputs = peft_model.generate(
    inputs,
    max_new_tokens=32,
    do_sample=False,           # 우선 greedy로 깔끔하게
    eos_token_id=eos
)

# 프롬프트 부분을 잘라 답변만 출력
gen_only = outputs[0, inputs.shape[-1]:]
print(tokenizer.decode(gen_only, skip_special_tokens=True).strip())

# 다른 질문 해보기 바다는 왜 파란가요?

messages = [
    {"role": "system", "content": "한국어로 짧고 정확하게만 답하세요. 모르면 모른다고 해라"},
    {"role": "user", "content": "바다는 왜 파란가요?"}
]

inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(peft_model.device)

outputs = peft_model.generate(
    inputs,
    max_new_tokens=32,
    do_sample=False,
    eos_token_id=eos
)

# 프롬프트 부분을 잘라 답변만 출력
gen_only = outputs[0, inputs.shape[-1]:]
print(tokenizer.decode(gen_only, skip_special_tokens=True).strip())


# 과적합체크

messages = [
    {"role": "system", "content": "한국어로 짧고 정확하게만 답하세요. 모르면 모른다고 해라"},
    {"role": "user", "content": "옆집 강아지 이름은?"}
]

inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(peft_model.device)

outputs = peft_model.generate(
    inputs,
    max_new_tokens=32,
    do_sample=False,
    eos_token_id=eos
)

# 프롬프트 부분을 잘라 답변만 출력
gen_only = outputs[0, inputs.shape[-1]:]
print(tokenizer.decode(gen_only, skip_special_tokens=True).strip())



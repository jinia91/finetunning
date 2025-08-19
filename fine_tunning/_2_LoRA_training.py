import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType  # LoRA 기반 파인튜닝 지원
import numpy as np

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id)


# 입력 문장을 숫자로 바꿔주는 tokenizer를 사용해 모델이 이해할 수 있는 형식으로 변환합니다.
# return_tensors="pt"는 PyTorch 텐서 형태로 반환하겠다는 뜻입니다.

lora_config = LoraConfig(
    r=16,  # 랭크: LoRA 내부 차원. 작을수록 가볍고 빠름
    lora_alpha=8,  # 학습 안정성을 위한 스케일링 계수
    target_modules=["c_attn","c_proj","q_attn"],  # GPT-2에서 LoRA를 적용할 레이어. 'c_attn'은 Attention 모듈의 핵심 부분입니다.
    # 🔍 GPT-2의 레이어들
    # - "c_attn": 쿼리, 키, 밸류를 생성하는 핵심 어텐션 입력 레이어
    # - "c_proj": 어텐션 출력 벡터를 변환하는 투영 레이어
    # - "q_attn": self-attention 중 쿼리 연산에 관여하는 부분 (GPT-J 등 일부 구조에 존재)
    # - "mlp.c_fc": 피드포워드 네트워크의 첫 번째 선형 계층 (MLP의 입력 부분)
    # - "mlp.c_proj": MLP의 출력 부분
    lora_dropout=0.05,  # 드롭아웃 적용 (과적합 방지)
    bias="none",  # bias 파라미터는 학습하지 않음
    task_type=TaskType.CAUSAL_LM,  # 작업 유형: 언어 생성 (Causal Language Modeling)
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

data = {
    "text": [
        "### 질문: 우리집 강아지 이름은?\n### 답변: 순둥이",
        "### 질문: 바다는 왜 파란가요?\n### 답변: 햇빛의 산란",
    ]
}
dataset = Dataset.from_dict(data)

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)
tokenized_dataset = dataset.map(tokenize_function)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir="../results",  # 결과 디렉토리
    per_device_train_batch_size=10,  # GPU나 CPU 1개당 배치 사이즈
    num_train_epochs=50,  # 데이터 전체를 몇 번 반복해서 학습할지 (여기선 10번)
    logging_steps=1,  # 1스텝마다 로그 출력 (학습 진행 상황 확인)
    save_strategy="no",  # 중간 저장 생략 (데모 목적이므로)
    fp16=False,  # GPU 없이 CPU로 학습 시 False 설정,
         report_to = "none",  # wandb, tensorboard 등 외부 로깅 툴 비활성화
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # padding 부분은 무시
    mask = labels != -100
    correct = (predictions == labels) & mask
    accuracy = correct.sum() / mask.sum()

    return {"accuracy": accuracy}
trainer = Trainer(
    model=model,  # 학습할 모델
    args=training_args,  # 학습 설정
    train_dataset=tokenized_dataset,  # 학습 데이터
    tokenizer=tokenizer,  # 텍스트 디코딩용
    data_collator=data_collator,  # 배치 구성 도우미
    compute_metrics=compute_metrics  # <-- 여기 추가

)

# 학습 시작!
trainer.train()


# "우리집 강아지 이름은?"이라는 질문에 답을 하도록 구성합니다.
input_text = "### 질문:우리집 강아지 이름은?\n### 답변:"

# 입력 문장을 숫자로 바꿔주는 tokenizer를 사용해 모델이 이해할 수 있는 형식으로 변환합니다.
# return_tensors="pt"는 PyTorch 텐서 형태로 반환하겠다는 뜻입니다.
inputs = tokenizer(input_text, return_tensors="pt")

# 모델이 올라가 있는 디바이스(GPU 또는 CPU)를 가져옵니다.
device = model.device

# 입력 데이터도 모델이 있는 디바이스로 옮겨줍니다.
# 그래야 모델과 데이터가 같은 장치에 있어 연산이 가능합니다.
inputs = {k: v.to(device) for k, v in inputs.items()}

# 모델에게 답변을 생성하도록 지시합니다.
# max_new_tokens=50은 최대 50개의 새로운 단어(토큰)를 생성하겠다는 의미입니다.
outputs = model.generate(**inputs, max_new_tokens=50)

# 생성된 답변을 사람이 읽을 수 있는 문자열로 바꿔서 출력합니다.
# skip_special_tokens=True는 시작/종료 같은 특수 기호는 출력하지 않겠다는 의미입니다.
print(tokenizer.decode(outputs[0], skip_special_tokens=True))




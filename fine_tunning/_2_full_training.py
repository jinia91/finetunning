import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# STEP 3: 모델 및 토크나이저 로드
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id)
model.resize_token_embeddings(len(tokenizer))  # pad_token 추가 고려

# STEP 4: 학습 데이터 정의
data = {
    "text": [
        "### 질문: 우리집 강아지 이름은?\n### 답변: 순둥이",
        "### 질문: 바다는 왜 파란가요?\n### 답변: 햇빛의 산란",
    ]
}
dataset = Dataset.from_dict(data)

# STEP 5: 토큰화
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function)

# STEP 6: 데이터 콜레이터
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# STEP 7: 트레이닝 설정
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=10,
    num_train_epochs=50,
    logging_steps=1,
    save_strategy="no",
    fp16=False,
         report_to = "none",  # wandb, tensorboard 등 외부 로깅 툴 비활성화

)

import numpy as np

# STEP 8: 트레이너 설정 및 학습


# 정확도를 계산하는 함수 정의
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # padding 부분은 무시
    mask = labels != -100
    correct = (predictions == labels) & mask
    accuracy = correct.sum() / mask.sum()

    return {"accuracy": accuracy}
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics # <-- 여기 추가

)

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





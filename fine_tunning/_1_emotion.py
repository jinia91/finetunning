from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score


# 1. 데이터셋 로드
dataset = load_dataset("imdb", split="train[:100]").train_test_split(test_size=0.2)
sample = dataset["train"][7]
# {'text': "Luise Rainer received an Oscar for her performance in The Good Earth. Unfortunately, her role required no. She did not say much and looked pale throughout the film. Luise's character was a slave then given away to marriage to Paul Muni's character (he did a fantastic job for his performance). Set in ancient Asia, both actors were not Asian, but were very convincing in their roles. I hope that Paul Muni received an Oscar for his performance, because that is what Luise must have gotten her Oscar for. She must have been a breakthrough actress, one of the first to method act. This seems like something that Hollywood does often. Al Pacino has played an Italian and Cuban. I felt Luise's performance to be lackluster throughout, and when she died, she did not change in expression from any previous scenes. She stayed the same throughout the film; she only changed her expression or emotion maybe twice. If her brilliant acting was so subtle, I suppose I did not see it.", 'label': 0}
print(sample)
print(f"📝 리뷰 내용:\n{sample['text']}\n")
print(f"🏷️ 라벨 (0=부정, 1=긍정): {sample['label']}")


# ✅ 2. 모델과 토크나이저 로드
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

dataset = dataset.map(tokenize, batched=True)


# ✅ 4. 훈련 설정 및 Trainer 구성
# 정확도 계산 함수
print("Setting up training arguments and trainer...")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

args = TrainingArguments(
    output_dir="test",  # 모델이 저장될 경로
    per_device_train_batch_size=8,  # 학습 시 배치 크기 (GPU 하나당 8개씩 처리)
    num_train_epochs=15,  # 에폭 수 (데이터셋을 15번 반복해서 학습)
     report_to = "none",  # wandb, tensorboard 등 외부 로깅 툴 비활성화
        logging_steps=1                   # 💡 몇 step마다 로그 출력할지
)
trainer = Trainer(
    model=model,  # 사용할 모델
    args=args,  # 훈련 설정값
    train_dataset=dataset["train"],  # 학습용 데이터셋
    eval_dataset=dataset["test"],  # 검증용 데이터셋
    compute_metrics=compute_metrics  # 💡 정확도 계산 함수 추가
)

# ✅ 5. 파인튜닝 실행 (모델 학습)
print("Starting training...")
trainer.train()

# ✅ 6. 모델 평가
print("Evaluating model...")
results = trainer.evaluate()
print(results)

# ✅ 7. 모델 저장
print("Saving model...")
model.save_pretrained("fine_tuned_model")

# 1. 필요한 라이브러리를 가져옵니다.
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# 2. 모델 경로와 기본 모델 이름을 지정합니다.
model_directory = "./fine_tuned_model"
base_model_name = "distilbert-base-uncased" # 파인튜닝에 사용된 기본 모델

print(f"파인튜닝된 모델 디렉토리: {os.path.abspath(model_directory)}")
print(f"기본 모델 (토크나이저용): {base_model_name}")

# 3. 모델과 토크나이저를 로드합니다.
try:
    # 토크나이저는 기본 모델에서 직접 로드합니다.
    print("토크나이저를 로드하는 중...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    print("토크나이저 로드 성공.")

    # 모델은 파인튜닝된 로컬 디렉토리에서 로드합니다.
    print("모델을 로드하는 중... (model.safetensors 사용)")
    model = AutoModelForSequenceClassification.from_pretrained(model_directory)
    print("모델 로드 성공.")

except Exception as e:
    print(f"모델 또는 토크나이저 로드 중 오류 발생: {e}")
    exit()

# 4. 분류를 원하는 샘플 텍스트를 준비합니다.
text_to_classify = "i dont like this movie, it was terrible and boring"

# 5. 토크나이저를 사용하여 텍스트를 모델 입력 형식으로 변환합니다.
inputs = tokenizer(text_to_classify, return_tensors="pt", truncation=True, padding=True)

print(f"\n분류할 텍스트: '{text_to_classify}'")

# 6. 모델을 사용하여 추론을 실행합니다.
# torch.no_grad()는 기울기 계산을 비활성화하여 메모리 사용량을 줄이고 계산 속도를 높입니다.
with torch.no_grad():
    outputs = model(**inputs)

# 7. 모델의 출력(logits)에서 가장 높은 값을 가진 클래스를 예측 결과로 선택합니다.
logits = outputs.logits
predicted_class_id = torch.argmax(logits, dim=-1).item()

# 8. 최종 결과를 출력합니다.
# 8. 최종 결과를 출력합니다.
if (predicted_class_id == 0):
    sentiment = "부정적"
else:
    sentiment = "긍정적"

print("반응 감정:", sentiment)

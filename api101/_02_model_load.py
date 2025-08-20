from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token
prefer_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if prefer_bf16 else torch.float16,
    device_map="auto",
)
model.eval()

user_text = "안녕하세요"
messages = [
    {"role": "system", "content": "당신은 간결하고 공손한 한국어 비서입니다."},
    {"role": "user", "content": user_text},
]

try:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
except Exception:
    prompt = user_text

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# 단순히 모델에 넘기면 logits가 반환됨
# logits이란 모델의 출력값으로, 각 토큰에 대한 확률 분포를 나타낸다
# generate는 이 분포표를 통해 실제로 다음 토큰들을 선택해 문장을 만들어내는 과정
output = model(inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
print(output)

with torch.no_grad():
    for step in range(output.logits.shape[1]):
        step_logits = output.logits[:, step, :]
        probs = torch.softmax(step_logits, dim=-1)
        topk = torch.topk(probs, k=10, dim=-1)

        topk_ids = topk.indices[0].tolist()
        topk_probs = topk.values[0].tolist()
        topk_tokens = tokenizer.convert_ids_to_tokens(topk_ids)
        topk_texts = [tokenizer.decode([tid]) for tid in topk_ids]

        print(f"=== Step {step} Top-k 후보 토큰 ===")
        for tok, text, p in zip(topk_tokens, topk_texts, topk_probs):
            print(f"{tok:>12s} ({text}) : {p:.4f}")
        print()

with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

new_tokens = generated[0, inputs["input_ids"].shape[1]:]
output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("=== MODEL REPLY ===")
print(output_text.strip())
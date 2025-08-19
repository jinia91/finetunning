import transformers
import torch

model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.float16,   # bfloat16 -> float16
        "low_cpu_mem_usage": True,
    },
    device_map="auto",
)
pipeline.model.eval()

messages = [
    {"role": "system", "content": "한국어로 짧고 정확하게만 답하세요."},
    {"role": "user", "content": "우리집 강아지 이름은?"}
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.1,
    top_p=0.9
)

print(outputs[0]["generated_text"][len(prompt):])

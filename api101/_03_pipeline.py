from transformers import pipeline

model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
pipe = pipeline(
    model=model_id,
    task="text-generation",
)

messages = [
    {"role": "system", "content": "너는 한국어를 영어로 번역하는 번역기인데, 반드시 먼저 step-by-step으로 추론 과정을 자세히 설명하고 마지막에 'Final Answer:' 뒤에 최종 번역만 출력해."},
    {"role": "user", "content": "안녕하세요, 오늘 날씨가 어때요? 영어로 번역해봐"},
]

prompt = pipe.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

outputs = pipe(
    prompt,
    max_new_tokens=500,
    do_sample=False,
    temperature=0.3,
    top_p=0.9,
    num_return_sequences=1,
    eos_token_id=[pipe.tokenizer.eos_token_id, pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    pad_token_id=pipe.tokenizer.eos_token_id,
    return_full_text=False,
)

print(outputs[0]["generated_text"].strip())
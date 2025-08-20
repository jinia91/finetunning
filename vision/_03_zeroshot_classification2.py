import torch
from PIL import Image
import requests
from datasets import load_dataset

from transformers import CLIPProcessor, CLIPModel

model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)
model.eval()

dataset = load_dataset("sasha/dog-food")
images = dataset["test"]["image"][:10]
labels = ["dog", "food"]
inputs = processor(images=images, text=labels, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    print("outputs:", outputs.keys())
    print("logits_per_image:", logits_per_image)
    print("probs:", probs)

for idx, prob in enumerate(probs):
    print(f"image # {idx + 1}")
    for label, score in zip(labels, prob):
        print(f"  {label}: {score:.4f}")
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = "https://i.namu.wiki/i/3wv4VNl3-SNQxQIAhkpQ_x1pFPRzyxK3tWlfdbB4sEBTe6ej0IKgcS5XT57b7nG_RIEDcutZPS72oXKYHAIRq1uJMkgfaPw0hewQY82SDkQM1psd-rBndjJfcx_D7EmNO3_ZQX3hF8D-ER4BNF_97Q.webp"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of paper", "another"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
scores = probs.tolist()

print("Scores:", scores)


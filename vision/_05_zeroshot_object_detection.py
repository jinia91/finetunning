import torch
from PIL import ImageDraw, ImageFont
from datasets import load_dataset
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import matplotlib.pyplot as plt

# 초기화 및 전체 모델과 데이터세트 로드
model_name = "google/owlv2-base-patch16"
processor = Owlv2Processor.from_pretrained(model_name)
model = Owlv2ForObjectDetection.from_pretrained(model_name)
dataset = load_dataset("francesco/animals-ij5d2")
print(dataset)
print(dataset["test"][0])


# Owlv2 프로세서를 통한 전처리
images = dataset["test"]["image"][:20]
categories = dataset["test"].features["objects"]["category"].feature.names
labels = [categories] * len(images)
inputs = processor(images=images, text=labels, return_tensors="pt", padding=True)

print(images)
print(labels)
print("input_ids:", inputs["input_ids"])
print("attention_mask:", inputs["attention_mask"])
print("pixel_values:", inputs["pixel_values"])
print("shape:", inputs["pixel_values"].shape)

# 모델 추론
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

print(outputs.keys())
print("logits:", outputs.logits)
print("objectness:", outputs.objectness_logits.shape)
print("pred_boxes:", outputs.pred_boxes.shape)
print("class_embeds:", outputs.class_embeds.shape)

shape = [dataset["test"][:20]["width"], dataset["test"][:20]["height"]]
target_sizes = list(map(list, zip(*shape)))
detection = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.5
)
print("detection:", detection)

for idx, (img, detection) in enumerate(zip(images, detection)):
    im = img.copy()
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=50)

    for box, score, label in zip(detection["boxes"], detection["scores"], detection["labels"]):
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=3)

    label_text = f"{labels[idx][label]} : {round(score.item(), 3)}"
    draw.text(box, label_text, fill="red", font=font)
    plt.imshow(im)
    plt.axis("off")
    plt.show()
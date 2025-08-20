from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
from PIL import ImageDraw, ImageFont
import requests
import matplotlib.pyplot as plt

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# Draw boxes and scores on the image
draw = ImageDraw.Draw(image)
try:
    # Pillow >= 10.0 supports size kwarg
    font = ImageFont.load_default(size=16)
except TypeError:
    # Older Pillow
    font = ImageFont.load_default()

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    # rectangle expects [left, top, right, bottom]
    draw.rectangle(box, outline="red", width=3)
    caption = f"{model.config.id2label[label.item()]}: {score.item():.3f}"
    # Text background for readability
    text_w, text_h = draw.textbbox((0, 0), caption, font=font)[2:]
    text_bg = [box[0], box[1] - text_h - 4, box[0] + text_w + 4, box[1]]
    draw.rectangle(text_bg, fill="red")
    draw.text((box[0] + 2, box[1] - text_h - 2), caption, fill="white", font=font)

# Show the image with matplotlib
plt.imshow(image)
plt.axis("off")
plt.tight_layout()
plt.show()

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
    )



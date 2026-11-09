import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import requests

# Load pretrained ResNet18 with default weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

# Load a local image (make sure 'dog.jpg' is in the same folder)
img = Image.open("dog.jpg").convert("RGB")

# Preprocess the image for ResNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
input_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

# Run inference
with torch.no_grad():
    output = model(input_tensor)

# Get predicted class index
_, predicted_class = output.max(1)
class_index = predicted_class.item()

# Load ImageNet class labels
class_index_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
class_names = requests.get(class_index_url).text.strip().split("\n")

# Print result
print(f"Predicted class index: {class_index}")
print(f"Predicted label: {class_names[class_index]}")

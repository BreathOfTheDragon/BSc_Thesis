# <editor-fold desc="Importing Libraries">
from torchvision import models
from torchvision.models import resnet18
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
# </editor-fold>


res18 = models.resnet18(pretrained=True)


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

image = Image.open("C:\\Users\\Asus\\Desktop\\dog2.jpg")
image_t = transform(image)
batch_t = torch.unsqueeze(image_t, 0)

res18.eval()

out = res18(batch_t)
with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

_, indices = torch.sort(out, descending=True)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print([(classes[idx], percentage[idx].item()) for idx in indices[0][:5]])


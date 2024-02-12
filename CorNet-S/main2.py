import torch
from torchvision import transforms
import thingsvision.vision as vision
from thingsvision.model_class import Model



mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

device = 'cuda'
source = 'torchvision'
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

model = Model(model_name='cornet-s', pretrained=True, device=device, source=source)

dl = vision.load_dl(root='C:/Users/Asus/Desktop/Project/Images',
                    out_path='C:/Users/Asus/Desktop/Project/CorNet-S/A train nodes',
                    transforms=transform)

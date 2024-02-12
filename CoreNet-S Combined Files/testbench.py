# <editor-fold desc="Importing Libraries">
import os
import json
import torch
import pickle
import cornet
import tifffile
import cv2 as cv
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from pydoc import importfile
from torchvision import models
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
import torchvision.datasets as dataset
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor

# </editor-fold>
# <editor-fold desc="Description">
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
k = 0
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])
transform_show_RGB = transforms.Compose([
    transforms.ToPILImage(mode='RGB')
])
images_directory = "C:\\Users\\Asus\\Desktop\\Project\\Images"
images = []  # a list of all images
images_arr = []  # a list of all images in arr format
image_names = []  # a list of all image names in order
image_names_notif = []  # a list of image names without .tif extension
images_transformed = []  # a list of the result of all transformed images
images_transformed_un_squeezed = []  # a list of the result of all transformed and un squeezed images
image_transformed_shown_in_RGB = []  # a list of the result of all transformed images in RGB
for file in os.listdir(images_directory):
    if file != 'desktop.ini':
        image = tifffile.imread(images_directory + "\\" + file)
        image = Image.fromarray(image)
        images.append(image)
        images_arr.append(tifffile.imread(images_directory + "\\" + file))
        image_names.append(file)
for i in range(len(images)):
    images_transformed.append(transform(images[i]))
    images_transformed_un_squeezed.append(images_transformed[i].unsqueeze(0))
    image_names_notif.append(os.path.splitext(image_names[i])[0])
# </editor-fold>


cornet_s = cornet.cornet_s(pretrained=True)
res18 = models.resnet18(pretrained=True)

print(cornet_s.named_modules())
cornet_s.eval()
cornet_s(images_transformed_un_squeezed[0])
#print(cornet_s.test(layer='decoder', sublayer='avgpool')._store_feats)


import os
import sys
import json
import torch
import pickle
import tifffile
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
import thingsvision.vision
from thingsvision.model_class import Model
import torchvision.datasets as dataset
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
import cornet

cornet_s = cornet.cornet_s(pretrained=True)
print(cornet_s)
slice_names = ['V1.conv1',
               'V1.norm1',
               'V1.nonlin1',
               'V1.pool',
               'V1.conv2',
               'V1.norm2',
               'V1.nonlin2',
               'V1.output',
               'V2.conv_input',
               'V2.skip',
               'V2.norm_skip',
               'V2.conv1',
               'V2.nonlin1',
               'V2.conv2',
               'V2.nonlin2',
               'V2.conv3',
               'V2.nonlin3',
               'V2.output',
               'V2.norm1_0',
               'V2.norm2_0',
               'V2.norm3_0',
               'V2.norm1_1',
               'V2.norm2_1',
               'V2.norm3_1',
               'V4.conv_input',
               'V4.skip',
               'V4.norm_skip',
               'V4.conv1',
               'V4.nonlin1',
               'V4.conv2',
               'V4.nonlin2',
               'V4.conv3',
               'V4.nonlin3',
               'V4.output',
               'V4.norm1_0',
               'V4.norm2_0',
               'V4.norm3_0',
               'V4.norm1_1',
               'V4.norm2_1',
               'V4.norm3_1',
               'V4.norm1_2',
               'V4.norm2_2',
               'V4.norm3_2',
               'V4.norm1_3',
               'V4.norm2_3',
               'V4.norm3_3',
               'IT.conv_input',
               'IT.skip',
               'IT.norm_skip',
               'IT.conv1',
               'IT.nonlin1',
               'IT.conv2',
               'IT.nonlin2',
               'IT.conv3',
               'IT.nonlin3',
               'IT.output',
               'IT.norm1_0',
               'IT.norm2_0',
               'IT.norm3_0',
               'IT.norm1_1',
               'IT.norm2_1',
               'IT.norm3_1',
               'decoder.avgpool',
               'decoder.flatten',
               'decoder.linear',
               'decoder.output'
               ]

print(slice_names)
print(len(slice_names))
model_name = 'cornet-s'
source = 'torchvision'
device = 'cuda'
batch_size = 1
model = Model(model_name, pretrained=True, device=device, source=source)

for i in range(len(slice_names)):
    dl = thingsvision.vision.load_dl(
        root='C:\\Users\\Asus\\Desktop\\Project\\Images\\',
        out_path=f'./{model_name}/' + str(i + 1) + f'_{slice_names[i]}/{slice_names[i]}',
        batch_size=batch_size,
        transforms=model.get_transformations(),
        backend=model.backend,
    )
    features, targets = model.extract_features(
        data_loader=dl,
        module_name=slice_names[i],
        flatten_acts=False,
        clip=False,
    )

    # features = vision.center_features(features)
    # features = vision.normalize_features(features)

    thingsvision.vision.save_features(features, out_path="C:\\Users\\Asus\\Desktop\\Project\\CorNet-S\\A train nodes\\",
                                      file_format='npy')

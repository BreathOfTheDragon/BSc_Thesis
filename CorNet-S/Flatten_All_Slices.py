import os
import re
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
import torchvision.datasets as dataset
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
import cornet

images_directory = "C:\\Users\\Asus\\Desktop\\Project\\Images"
A_directory = 'C:\\Users\\Asus\\Desktop\\Project\\CorNet-S\\A train nodes\\'

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
image_names = []
image_names_notif = []
image_by_complex_slice_size = []
image_by_complex_slice_size_npy = []
file_names = []
for file in os.listdir(images_directory):
    if file == 'desktop.ini':
        pass
    else:
        image_names.append(file)
for b in range(len(image_names)):
    image_names_notif.append(os.path.splitext(image_names[b])[0])

for file in os.listdir(A_directory):
    if file == 'desktop.ini':
        pass
    else:
        image_by_complex_slice_size_npy.append(file)
        file_names.append(file)
        print(file)

for kkk in range(len(image_by_complex_slice_size_npy)):
    file_name, file_extension = os.path.splitext(A_directory + image_by_complex_slice_size_npy[kkk])
    image_by_complex_slice_size.append(np.load(A_directory + image_by_complex_slice_size_npy[kkk]))
    print(image_by_complex_slice_size[kkk].shape)
    image_by_slice_stacked = np.reshape(image_by_complex_slice_size[kkk], newshape=(-1, 155))
    print(image_by_slice_stacked.shape)
    pickle.dump(image_by_slice_stacked,
                open('C:\\Users\\Asus\\Desktop\\Project\\CorNet-S\\CNN Slices Flattened by Image as pickle\\' +
                     os.path.splitext(file_names[kkk])[0] +
                     '_for_all_images.p', "wb"))

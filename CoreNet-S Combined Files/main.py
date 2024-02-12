import os
import json
import torch
import pickle
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
import cornet
res18 = models.resnet18(pretrained=True)
cornet_s = cornet.cornet_s(pretrained=True)
# print(my_cornet)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
k = 0
transform = transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    transforms.Normalize(mean, std)])

transform_show_RGB = transforms.Compose([
    transforms.ToPILImage(mode='RGB')
])
print(cornet_s.__class__)
images_directory = "E:\\Project\\Images"
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




print(cornet_s.module)



# slice_names = np.array([])  # a list of the names of all slices of the network
# slice_names = np.append(slice_names, np.array(models.feature_extraction.get_graph_node_names(cornet_s.module())[0]))
# print(slice_names)
#
# all_images_all_layers = dict.fromkeys(image_names_notif, None)
#
# i = 0
# for i in range(len(images)):
#     slice_outputs = dict.fromkeys(slice_names, None)
#     cornet_s.eval()
#     cornet_s(images_transformed_un_squeezed[i])
#     j = 0
#     for j in range(slice_names.size):
#         slice_output = create_feature_extractor(cornet_s, return_nodes=[slice_names[j]])
#         with torch.no_grad():
#             slice_outputs[slice_names[j]] = slice_output(images_transformed_un_squeezed[i])[slice_names[j]]
#     all_images_all_layers[image_names_notif[i]] = slice_outputs
#     k += 1
#     print(k)
#     pickle.dump(all_images_all_layers[image_names_notif[i]],
#                 open("C:\\Users\\Asus\\Desktop\\Project\\CorNet-S\\CNN Layer outputs in tensor as pickle\\" +
#                      image_names_notif[
#                          i] + '_tensor.p', "wb"))

# conv_slices = []
# conv_slice_shapes = []
# conv_slice_indices = []
# conv_phrase = 'conv'
# x_phrase = 'x'
#
# for x in range(len(slice_names)):
#     if slice_names[x] == x_phrase:
#         conv_slices.append(slice_names[x])
#         conv_slice_indices.append(x)
# for y in range(len(slice_names)):
#     if conv_phrase in slice_names[y]:
#         conv_slices.append(slice_names[y])
#         conv_slice_indices.append(y)
# for Q in range(len(images)):
#     conv_slices_data = []
#     conv_slices_data_squeezed = []
#     image_dictionary = pd.read_pickle(
#         r'C:\\Users\\Asus\\Desktop\\Project\\CorNet-S\\CNN Layer outputs in tensor as pickle\\' + image_names_notif[
#             Q] + '_tensor.p')
#     for p in range(len(conv_slices)):
#         conv_slice_shapes.append(image_dictionary[conv_slices[p]].shape)
#         conv_slices_data.append(image_dictionary[conv_slices[p]])
#     image_transformed_squeezed = []
#     processed = []
#     for fm in conv_slices_data:
#         fm = fm.squeeze(0)
#         gray_scale = torch.sum(fm, 0)
#         gray_scale = gray_scale / fm.shape[0]
#         processed.append(gray_scale.data.cpu().numpy())
#     for f in processed:
#         fig = plt.figure(figsize=(30, 50))
#     for i in range(len(processed)):
#         a = fig.add_subplot(5, 4, i + 1)
#         plt.imshow(processed[i])
#         a.axis("off")
#         a.set_title(conv_slices[i].split('(')[0], fontsize=45)
#     plt.savefig('C:\\Users\\Asus\\Desktop\\Project\\CorNet-S\\Images feature maps\\' + image_names_notif[Q] + '.png',
#                 bbox_inches='tight')
#     plt.close()
#

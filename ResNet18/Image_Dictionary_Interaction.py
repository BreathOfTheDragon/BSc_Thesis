import ast
import json
import os
import pickle
import numpy as np
import pandas as pd
from torchvision import models

np.set_printoptions(linewidth=200)
np.set_printoptions(threshold=np.inf)
res18 = models.resnet18(pretrained=True)

images_directory = "C:\\Users\\Asus\\Desktop\\Project\\Images"

slice_names = np.array([])  # a list of the names of all slices of the network
slice_names = np.append(slice_names, np.array(models.feature_extraction.get_graph_node_names(res18)[0]))
image_names = []

for file in os.listdir(images_directory):
    image_names.append(file)
width = 17

for i in range(int(len(image_names) / width)):
    print(image_names[i * width:(i + 1) * width])
print(
    "#################################################################################################################")
image_name = input("Enter the image name :")
image_name = image_name.upper()
image_dictionary = pd.read_pickle(
    r'C:\\Users\\Asus\\Desktop\\Project\\ResNet18\\CNN Layer outputs in tensor as pickle\\' + image_name + '_tensor.p')
print(slice_names)
print(
    "#################################################################################################################")
slice_name = input("Enter the slice name :")
print("shape is :", image_dictionary[slice_name].shape)
print("values is :", image_dictionary[slice_name])

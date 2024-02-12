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

print(slice_names)
user_slice_name = input("Enter the slice name :")

index = slice_names.tolist().index(user_slice_name)
print(index)

image_dictionary = pd.read_pickle(
    r'C:\\Users\\Asus\\Desktop\\Project\\ResNet18\\CNN Slices Flattened by Image as pickle\\' + str(
        index) + '_' + user_slice_name + '_for_all_images.p')

print(
    "#################################################################################################################")
print(type(image_dictionary))
print("shape is :", image_dictionary.shape)
print("values is :", image_dictionary[index])

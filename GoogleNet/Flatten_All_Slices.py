import ast
import json
import os
import pickle
import numpy as np
import pandas as pd
from torchvision import models

np.set_printoptions(linewidth=200)
np.set_printoptions(threshold=np.inf)
googlenet = models.googlenet(pretrained=True)

images_directory = "C:\\Users\\Asus\\Desktop\\Project\\Images"
slice_names = np.array([])  # a list of the names of all slices of the network
slice_names = np.append(slice_names, np.array(models.feature_extraction.get_graph_node_names(googlenet)[0]))
image_names = []
image_names_notif = []

for file in os.listdir(images_directory):
    if file == 'desktop.ini':
        pass
    else:
        image_names.append(file)
for b in range(len(image_names)):
    image_names_notif.append(os.path.splitext(image_names[b])[0])
print(image_names)
print(len(image_names))
print(image_names_notif)
print(slice_names)
for j in range(len(slice_names)):
    print("slice", j)
    k = 0
    image_by_slice = []
    image_by_slice_stacked = []
    for i in range(len(image_names_notif)):
        image_dictionary = pd.read_pickle(
            r'C:\\Users\\Asus\\Desktop\\Project\\GoogleNet\\CNN Layer outputs in tensor as pickle\\' + image_names_notif
            [
                i] + '_tensor.p')
        image_by_slice.append(image_dictionary[slice_names[j]].flatten())
        print("itteration", k)
        k += 1
    image_by_slice_stacked = np.stack(image_by_slice, axis=0)
    slice_by_image_stacked = image_by_slice_stacked.transpose()
    print(slice_by_image_stacked.shape)
    pickle.dump(slice_by_image_stacked,
                open("C:\\Users\\Asus\\Desktop\\Project\\GoogleNet\\CNN Slices Flattened by Image as pickle\\" + str(
                    j).zfill(3) + '_' +
                     slice_names[
                         j] + '_for_all_images.p', "wb"))

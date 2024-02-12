import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import rsatoolbox
import pickle
import sys
import os
import seaborn as sns
import scipy.spatial
import scipy.stats
import pandas as pd
import scipy
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor



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

for k in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-S\CNN Slices Flattened by Image as pickle\\' + str(k+1).zfill(2) + '_' +
        slice_names[
            k] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[74:155]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_matrix = sliceF_x_image
    comparisons = np.array([])
    slice_rdm = scipy.spatial.distance.pdist(slice_matrix, metric='euclidean')
    first = []
    second = []
    sum = 0
    sum2 = 79
    for r in range(slice_matrix.shape[0] - 1):
        first.append(sum)
        second.append(sum2)
        sum += 80 - r
        sum2 += 79 - r
    print(first)
    print(second)
    matrix = np.array([])
    rows = []
    print(slice_matrix.shape[0])
    for i in range(0, slice_matrix.shape[0]):
        row = np.array([])
        row = np.hstack((row, np.zeros(i + 1)))
        if i != (slice_matrix.shape[0] - 1):
            row = np.hstack((row, slice_rdm[first[i]: second[i] + 1]))
        elif i == (slice_matrix.shape[0] - 1):
            row = np.hstack(row)
        print(row)
        rows.append(row)

    matrix = np.stack(rows, axis=0)
    print(matrix)
    print(matrix.shape)

    for j in range(0, slice_matrix.shape[0]):
        for i in range(0, slice_matrix.shape[0]):
            if i != j:
                matrix[i][j] = matrix[j][i]
    print(matrix)
    print(matrix.shape)
    plt.figure(figsize=(20, 20))
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=10, rotation=90)
    plt.ylabel("heatmap", fontsize=25)
    ax = sns.heatmap(matrix, linewidth=0.5)
    plt.draw()
    plt.savefig(
        'C:\\Users\\Asus\\Desktop\\Project\\CorNet-S 2\\RDMs CNN all images plot\\' + str(k+1).zfill(2) + '_RDM_' +
        slice_names[k] + '_HIL_Images.png',
        dpi=300)
    plt.close()

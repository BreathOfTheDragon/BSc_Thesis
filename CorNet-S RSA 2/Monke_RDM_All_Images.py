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

k = 0
type_of_image = ['B']

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
monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monkey_matrix = monke_brain_matrix[74:155]

monke_brain_matrix = monkey_matrix
comparisons = np.array([])
monke_rdm = scipy.spatial.distance.pdist(monke_brain_matrix, metric='euclidean')

first = []
second = []
sum = 0
sum2 = 79
for r in range(monkey_matrix.shape[0] - 1):
    first.append(sum)
    second.append(sum2)
    sum += 80 - r
    sum2 += 79 - r
print(first)
print(second)

matrix = np.array([])
rows = []
print(monkey_matrix.shape[0])

for i in range(0, monkey_matrix.shape[0]):
    row = np.array([])
    row = np.hstack((row, np.zeros(i + 1)))
    if i != (monkey_matrix.shape[0] - 1):
        row = np.hstack((row, monke_rdm[first[i]: second[i] + 1]))
    elif i == (monkey_matrix.shape[0] - 1):
        row = np.hstack(row)
    print(row)
    rows.append(row)

matrix = np.stack(rows, axis=0)
print(matrix)
print(matrix.shape)

for j in range(0, monkey_matrix.shape[0]):
    for i in range(0, monkey_matrix.shape[0]):
        if i != j:
            matrix[i][j] = matrix[j][i]

print(matrix)
print(matrix.shape)

norm = np.linalg.norm(matrix)
normal_matrix = matrix / norm

plt.figure(figsize=(20, 20))
plt.yticks(fontsize=20)
plt.xticks(fontsize=10, rotation=90)
plt.ylabel("heatmap", fontsize=25)
# plt.title("Correlation between brain region and CorNet-S Slices", fontsize=25)
# plt.imshow(matrix, cmap='hot', interpolation='nearest')
# plt.show()
ax = sns.heatmap(matrix, linewidth=0.5)
plt.draw()
plt.savefig(
    'C:\\Users\\Asus\\Desktop\\Project\\CorNet-S 2\\RDMs Monke all images plot\\RDM_Monke_HIL_Images.png',
    dpi=100)
plt.close()

plt.figure(figsize=(20, 20))
plt.yticks(fontsize=20)
plt.xticks(fontsize=10, rotation=90)
plt.ylabel("heatmap", fontsize=25)
# plt.title("Correlation between brain region and CorNet-S Slices", fontsize=25)
# plt.imshow(matrix, cmap='hot', interpolation='nearest')
# plt.show()
ax = sns.heatmap(normal_matrix, linewidth=0.5)
plt.draw()
plt.savefig(
    'C:\\Users\\Asus\\Desktop\\Project\\CorNet-S 2\\RDMs Monke all images plot\\RDM_Normal_Monke_HIL_Images.png',
    dpi=100)
plt.close()

plt.figure(figsize=(20, 20))
plt.yticks(fontsize=20)
plt.xticks(fontsize=10, rotation=90)
plt.ylabel("heatmap", fontsize=25)
# plt.title("Correlation between brain region and CorNet-S Slices", fontsize=25)
# plt.imshow(matrix, cmap='hot', interpolation='nearest')
# plt.show()
ax = sns.heatmap(1 - normal_matrix, linewidth=0.5)
plt.draw()
plt.savefig(
    'C:\\Users\\Asus\\Desktop\\Project\\CorNet-S 2\\RDMs Monke all images plot\\RDM_1-Normal_Monke_HIL_Images.png',
    dpi=100)
plt.close()

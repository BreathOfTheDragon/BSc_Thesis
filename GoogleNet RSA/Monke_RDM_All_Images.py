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

googlenet = models.googlenet(pretrained=True)
slice_names = np.array([])  # a list of the names of all slices of the network
slice_names = np.append(slice_names, np.array(models.feature_extraction.get_graph_node_names(googlenet)[0]))
print(slice_names)

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monkey_matrix = monke_brain_matrix[74:155]

# <editor-fold desc="my calculation RDM">
comparisons = np.array([])
monke_rdm = scipy.spatial.distance.pdist(monkey_matrix, metric='euclidean')

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

plt.figure(figsize=(23, 20))
plt.yticks(fontsize=20, rotation=90)
plt.xticks(fontsize=20)
plt.ylabel("heatmap", fontsize=25)
# plt.title("Correlation between brain region and GoogleNet Slices", fontsize=25)
# plt.imshow(matrix, cmap='hot', interpolation='nearest')
# plt.show()
x_axis_labels = []
y_axis_labels = []
for hh in range(1, monkey_matrix.shape[0] + 1):
    if hh % 9 == 0:
        if hh <= 27:
            x_axis_labels.append('H' + str(hh))
            y_axis_labels.append('H' + str(hh))
        elif hh >= 28 and hh <= 54:
            x_axis_labels.append('I' + str(hh - 27))
            y_axis_labels.append('I' + str(hh - 27))
        elif hh >= 55 and hh <= 81:
            x_axis_labels.append('L' + str(hh - 54))
            y_axis_labels.append('L' + str(hh - 54))
    else:
        x_axis_labels.append(None)
        y_axis_labels.append(None)
ax = sns.heatmap(normal_matrix, xticklabels=x_axis_labels, yticklabels=y_axis_labels,
                 linewidth=0.5, cmap='afmhot')
plt.draw()
plt.savefig(
    'C:\\Users\\Asus\\Desktop\\Project\\GoogleNet\\RDMs Monke all images plot\\RDM_Monke_HIL_Images_mycalc.png',
    dpi=300)
plt.close()
# </editor-fold>

# <editor-fold desc="RSAtoolbox calculation RDM">
monke_data = rsatoolbox.data.Dataset(monkey_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)

rsatoolbox.vis.show_rdm(monke_rdm_non_square, figsize=(10, 8), show_colorbar='figure',
                        num_pattern_groups=27)
plt.draw()
plt.savefig('C:\\Users\\Asus\\Desktop\\Project\\GoogleNet\\RDMs Monke all images plot\\RDM_Monke_HIL_images_toolbox.png',
            dpi=300)
plt.close()
# </editor-fold>

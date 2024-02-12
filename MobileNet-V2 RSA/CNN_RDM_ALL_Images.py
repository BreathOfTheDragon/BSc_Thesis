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

mobilenetv2 = models.mobilenet_v2(pretrained=True)
slice_names = np.array([])  # a list of the names of all slices of the network
slice_names = np.append(slice_names, np.array(models.feature_extraction.get_graph_node_names(mobilenetv2)[0]))
print(slice_names)

# <editor-fold desc="my calculation RDM">
for k in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\MobileNet-V2\CNN Slices Flattened by Image as pickle\\' + str(k).zfill(
            3) + '_' +
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

    norm = np.linalg.norm(matrix)
    normal_matrix = matrix / norm

    plt.figure(figsize=(23, 20))
    plt.yticks(fontsize=20, rotation=90)
    plt.xticks(np.arange(0, slice_matrix.shape[0], 5), fontsize=20)
    plt.ylabel("heatmap", fontsize=25)
    x_axis_labels = []
    y_axis_labels = []
    for hh in range(1, sliceF_x_image.shape[0] + 1):
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
    sns.set(font_scale=2)
    ax = sns.heatmap(normal_matrix, xticklabels=x_axis_labels, yticklabels=y_axis_labels,
                     linewidth=0.5, cmap='afmhot')
    plt.draw()
    plt.savefig(
        'C:\\Users\\Asus\\Desktop\\Project\\MobileNet-V2\\RDMs CNN all images plot\\' + str(k).zfill(
            3) + '_RDM_' +
        slice_names[k] + '_HIL_Images_mycalc.png',
        dpi=300, bbox_inches='tight')
    plt.close()
# </editor-fold>

# # <editor-fold desc="RSAtoolbox calculation RDM">
# for w in range(len(slice_names)):
#     sliceF_x_image = pd.read_pickle(
#         r'C:\Users\Asus\Desktop\Project\MobileNet-V2\CNN Slices Flattened by Image as pickle\\' + str(w).zfill(3) + '_' +
#         slice_names[
#             w] + '_for_all_images' + '.p')
#     sliceF_x_image = sliceF_x_image.transpose()
#     sliceF_x_image = sliceF_x_image[74:155]
#     slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
#     slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
#     slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
#
#     rsatoolbox.vis.show_rdm(slice_rdm_non_square, figsize=(10, 8), show_colorbar='figure',
#                             num_pattern_groups=27)
#
#     plt.draw()
#     plt.savefig(
#         'C:\\Users\\Asus\\Desktop\\Project\\MobileNet-V2\\RDMs CNN all images plot\\' + str(w).zfill(3) + '_RDM_' +
#         slice_names[w] + '_HIL_Images_toolbox.png',
#         dpi=300)
#     # rsatoolbox.vis.show
#     plt.close()
# # </editor-fold>

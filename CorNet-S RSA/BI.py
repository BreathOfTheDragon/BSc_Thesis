import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import rsatoolbox
import pickle
import os
import pandas as pd
import scipy
from torchvision import models

k = 0
type_of_image = ['Human_Face_IF', 'Animal_Face_IF', 'Human_Body_IF', 'Animal_Body_IF', 'Natural_IF', 'Man_Made_IF',
                 'Face_IF', 'Body_IF',
                 'Animate_IF', 'Inanimate_IF', 'All_IF']

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
monke_brain_matrix = monke_brain_matrix[101:107]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-S\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[101:107]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-S/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[107:110]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-S\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[107:110]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-S/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[109:113]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-S\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[109:113]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-S/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[113:116]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-S\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[113:116]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-S/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[116: 122]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-S\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[116: 122]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-S/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[122:128]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-S\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[122:128]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-S/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[101:110]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-S\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[101:110]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-S/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[110:116]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-S\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[110:116]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-S/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[101:116]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-S\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[101:116]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-S/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[116: 128]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-S\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[116: 128]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-S/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[101: 128]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-S\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[101: 128]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-S/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

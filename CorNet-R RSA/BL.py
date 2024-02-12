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
type_of_image = ['Human_Face_LF', 'Animal_Face_LF', 'Human_Body_LF', 'Animal_Body_LF', 'Natural_LF', 'Man_Made_LF',
                 'Face_LF', 'Body_LF',
                 'Animate_LF', 'Inanimate_LF', 'All_LF']

slice_names = ['V1.conv_input',
               'V1.norm_input',
               'V1.nonlin_input',
               'V1.conv1',
               'V1.norm1',
               'V1.nonlin1',
               'V1.output',
               'V2.conv_input',
               'V2.norm_input',
               'V2.nonlin_input',
               'V2.conv1',
               'V2.norm1',
               'V2.nonlin1',
               'V2.output',
               'V4.conv_input',
               'V4.norm_input',
               'V4.nonlin_input',
               'V4.conv1',
               'V4.norm1',
               'V4.nonlin1',
               'V4.output',
               'IT.conv_input',
               'IT.norm_input',
               'IT.nonlin_input',
               'IT.conv1',
               'IT.norm1',
               'IT.nonlin1',
               'IT.output',
               'decoder.avgpool',
               'decoder.flatten',
               'decoder.linear'
               ]

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[128:134]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-R\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[128:134]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-R/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[134:137]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-R\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[134:137]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-R/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[137:140]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-R\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[137:140]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-R/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[140:143]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-R\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[140:143]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-R/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[143: 149]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-R\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[143: 149]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-R/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[149:155]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-R\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[149:155]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-R/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[128:137]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-R\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[128:137]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-R/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[137:143]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-R\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[137:143]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-R/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[128:143]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-R\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[128:143]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-R/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[143: 155]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-R\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[143: 155]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-R/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[128: 155]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-R\CNN Slices Flattened by Image as pickle\\' + str(i + 1).zfill(
            2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[128: 155]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-R/RDM Comparisons B mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[
        k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

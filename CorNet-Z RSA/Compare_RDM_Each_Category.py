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
type_of_image = ['1_HF noF', '2_AF noF', '3_HBP noF', '4_ABP noF', '5_N noF', '6_MM noF', '7_HF hF', '8_AF hF',
                 '9_HBP hF',
                 '10_ABP hF', '11_N hF', '12_MM hF', '13_HF iF', '14_AF iF', '15_HBP iF', '16_ABP iF', '17_N iF',
                 '18_MM iF',
                 '19_HF LF', '20_AF LF', '21_HBP LF', '22_ABP LF', '23_N LF', '24_MM LF', '25_All Images']
slice_names = ['V1.conv',
               'V1.nonlin',
               'V1.pool',
               'V1.output',

               'V2.conv',
               'V2.nonlin',
               'V2.pool',
               'V2.output',

               'V4.conv',
               'V4.nonlin',
               'V4.pool',
               'V4.output',

               'IT.conv',
               'IT.nonlin',
               'IT.pool',
               'IT.output',

               'decoder.avgpool',
               'decoder.flatten',
               'decoder.linear',
               'decoder.output'
               ]

# # <editor-fold desc="base code">
# k+=1
# monke_brain_matrix = monke_brain_matrix[:]
# comparisons = []
# monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
# monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
# monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
# for i in range(len(slice_names)):
#     sliceF_x_image = pd.read_pickle(
#         r'C:\Users\Asus\Desktop\Project\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2 + '_' + slice_names[
#             i] + '_for_all_images' + '.p')
#     sliceF_x_image = sliceF_x_image.transpose()
#     sliceF_x_image = sliceF_x_image[:]
#     slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
#     slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
#     slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
#     comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
#     print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
#     print(i)
# scipy.io.savemat(
#     'C:/Users/Asus/Desktop/Project/RDM Comparison mats/RDM_comparison_' + type_of_image[k] + '.mat',
#     {'RDM_comparison' + str(k): comparisons})
# # </editor-fold>

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[0:9]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[0:9]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1
monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[9:18]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[9:18]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1
monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[18:28]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[18:28]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1
monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[28:37]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[28:37]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1
monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[np.r_[37:54, 71:74]]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[np.r_[37:54, 71:74]]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1
monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[54:71]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[54:71]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1
monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[74:80]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[74:80]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1
monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[80:83]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[80:83]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1
monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[83:86]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[83:86]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1
monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[86:89]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[86:89]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1
monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[89:95]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[89:95]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1
monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[95:101]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[95:101]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1
monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[101:107]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
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
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
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
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
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
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1
monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[110:113]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[110:113]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
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
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
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
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1
monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[116:122]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[116:122]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
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
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
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
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1
monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[128:134]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
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
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
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
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
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
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
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
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
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
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
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
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
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
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1
monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[143:149]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[143:149]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
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
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
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
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

i = 0
k += 1

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
monke_brain_matrix = monke_brain_matrix[:]
comparisons = []
monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)
for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-Z\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    sliceF_x_image = sliceF_x_image[:]
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    comparisons.append(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))
    print(i)
scipy.io.savemat(
    'C:/Users/Asus/Desktop/Project/CorNet-Z/RDM Comparison mats/' + str(k + 1) + '_RDM_comparison_' + type_of_image[k] + '.mat',
    {'RDM_comparison' + str((k + 1)): comparisons})

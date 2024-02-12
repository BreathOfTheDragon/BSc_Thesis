import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import rsatoolbox
import pickle
import os
import pandas as pd
import scipy
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from scipy.interpolate import interp1d

my_models = ['AlexNet', 'GoogleNet', 'MobileNet-V2', 'ResNet18', 'ResNet50',
             'ResNet101',
             'SqueezeNet1_1',
             'VGG-16', 'VGG-19']
type_of_image = ['Face_HF', 'Face_IF', 'Face_LF', 'Body_HF', 'Body_IF', 'Body_LF']
color = ['red', 'blue', 'green']
torchmodels = [models.alexnet(pretrained=True), models.googlenet(pretrained=True), models.mobilenet_v2(pretrained=True),
               models.resnet18(pretrained=True), models.resnet50(pretrained=True), models.resnet101(pretrained=True),
               models.squeezenet1_1(pretrained=True), models.vgg16(pretrained=True), models.vgg19(pretrained=True)]

comparison_mats_directory = []
inx = 0
for model in my_models:
    if model != 'CorNet-S' and model != 'CorNet-R' and model != 'CorNet-Z':
        slice_names = np.array([])  # a list of the names of all slices of the network
        slice_names = np.append(slice_names,
                                np.array(models.feature_extraction.get_graph_node_names(torchmodels[inx])[0]))
        print(slice_names.shape)
        inx += 1
        comparison_mats_directory = "C:\\Users\\Asus\\Desktop\\Project\\" + model + "\\RDM Comparisons B mats\\"
        comparison_mats = []
        fl = [7, 7, 7, 8, 8, 8]
        for u in range(len(type_of_image)):
            comparison_mats.append(
                scipy.io.loadmat(comparison_mats_directory + str(fl[u]) + '_RDM_comparison_' + type_of_image[u]))
        big_values = []
        for l in range(len(comparison_mats)):
            comparisons = comparison_mats[l]['RDM_comparison']
            comparisons = comparisons.squeeze()
            print(comparisons.shape)
            data = dict.fromkeys(slice_names, None)
            for ui in range(len(slice_names)):
                data[slice_names[ui]] = comparisons[ui]

            slices = list(data.keys())
            values = list(data.values())
            print(slices)
            print(values)
            big_values.append(values)
        glist = [0, 3]
        types = ['Face', 'Body']
        oo = 0
        for g in glist:
            listtt = [-1, -0.5, 0, 0.5, 1]
            plt.figure(figsize=(30, 10))
            plt.yticks(listtt, fontsize=25)
            plt.ylim(-1.0, 1.0)
            # plt.legend(fontsize=15)
            # plt.xticks(np.arange(0, len(slices[1:]), 5), fontsize=0, rotation=90)
            plt.tick_params(axis='x', labelbottom=False)
            # plt.title(model, fontsize=30)
            plt.ylabel("Correlation", fontsize=25)

            plt.plot(slices[1:], big_values[g][1:], color=color[g % 3], linewidth=3, label=type_of_image[g])
            plt.plot(slices[1:], big_values[g + 1][1:], color=color[(g + 1) % 3], linewidth=3,
                     label=type_of_image[g + 1])
            plt.plot(slices[1:], big_values[g + 2][1:], color=color[(g + 2) % 3], linewidth=3,
                     label=type_of_image[g + 2])

            plt.legend(fontsize=22, loc=3)
            plt.draw()
            plt.savefig(
                'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models\\Body and Face plots\\' + model + "_" + types[
                    oo] + '.png', bbox_inches='tight',

                dpi=100)
            plt.close()
            oo += 1
    # elif model == 'CorNet-S':
    #     slice_names = ['V1.conv1',
    #                    'V1.norm1',
    #                    'V1.nonlin1',
    #                    'V1.pool',
    #                    'V1.conv2',
    #                    'V1.norm2',
    #                    'V1.nonlin2',
    #                    'V1.output',
    #                    'V2.conv_input',
    #                    'V2.skip',
    #                    'V2.norm_skip',
    #                    'V2.conv1',
    #                    'V2.nonlin1',
    #                    'V2.conv2',
    #                    'V2.nonlin2',
    #                    'V2.conv3',
    #                    'V2.nonlin3',
    #                    'V2.output',
    #                    'V2.norm1_0',
    #                    'V2.norm2_0',
    #                    'V2.norm3_0',
    #                    'V2.norm1_1',
    #                    'V2.norm2_1',
    #                    'V2.norm3_1',
    #                    'V4.conv_input',
    #                    'V4.skip',
    #                    'V4.norm_skip',
    #                    'V4.conv1',
    #                    'V4.nonlin1',
    #                    'V4.conv2',
    #                    'V4.nonlin2',
    #                    'V4.conv3',
    #                    'V4.nonlin3',
    #                    'V4.output',
    #                    'V4.norm1_0',
    #                    'V4.norm2_0',
    #                    'V4.norm3_0',
    #                    'V4.norm1_1',
    #                    'V4.norm2_1',
    #                    'V4.norm3_1',
    #                    'V4.norm1_2',
    #                    'V4.norm2_2',
    #                    'V4.norm3_2',
    #                    'V4.norm1_3',
    #                    'V4.norm2_3',
    #                    'V4.norm3_3',
    #                    'IT.conv_input',
    #                    'IT.skip',
    #                    'IT.norm_skip',
    #                    'IT.conv1',
    #                    'IT.nonlin1',
    #                    'IT.conv2',
    #                    'IT.nonlin2',
    #                    'IT.conv3',
    #                    'IT.nonlin3',
    #                    'IT.output',
    #                    'IT.norm1_0',
    #                    'IT.norm2_0',
    #                    'IT.norm3_0',
    #                    'IT.norm1_1',
    #                    'IT.norm2_1',
    #                    'IT.norm3_1',
    #                    'decoder.avgpool',
    #                    'decoder.flatten',
    #                    'decoder.linear',
    #                    'decoder.output']
    #     comparison_mats_directory = "C:\\Users\\Asus\\Desktop\\Project\\" + model + "\\RDM Comparisons B mats\\"
    #     comparison_mats = []
    #     for u in range(len(type_of_image)):
    #         comparison_mats.append(
    #             scipy.io.loadmat(comparison_mats_directory + 'RDM_comparison_' + type_of_image[u]))
    #     big_values = []
    #     for l in range(len(comparison_mats)):
    #         comparisons = comparison_mats[l]['RDM_comparison' + str(int(l / 3) + 7)]
    #         comparisons = comparisons.squeeze()
    #         data = dict.fromkeys(slice_names, None)
    #         for i in range(len(slice_names)):
    #             data[slice_names[i]] = comparisons[i]
    #         print(comparisons.shape)
    #         slices = list(data.keys())
    #         values = list(data.values())
    #         print(slices)
    #         print(values)
    #         big_values.append(values)
    #     glist = [0, 3]
    #     types = ['Face', 'Body']
    #     oo = 0
    #     for g in glist:
    #         plt.figure(figsize=(20, 20))
    #         plt.yticks(fontsize=20)
    #         plt.ylim(-1.0, 1.0)
    #         plt.legend(fontsize=15)
    #         plt.xticks(fontsize=10, rotation=90)
    #         plt.title(model, fontsize=25)
    #         plt.ylabel("Correlation", fontsize=25)
    #         plt.plot(slices, big_values[g], color=color[g % 3], linewidth=3, label=type_of_image[g])
    #         plt.plot(slices, big_values[g + 1], color=color[(g + 1) % 3], linewidth=3, label=type_of_image[g + 1])
    #         plt.plot(slices, big_values[g + 2], color=color[(g + 2) % 3], linewidth=3, label=type_of_image[g + 2])
    #         plt.legend(fontsize=30)
    #         plt.draw()
    #         plt.savefig(
    #             'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models\\Body and Face plots\\' + model + "_" + types[
    #                 oo] + '.png',
    #
    #             dpi=100)
    #         plt.close()
    #         oo += 1
    #
    # elif model == 'CorNet-R':
    #     slice_names = ['V1.conv_input',
    #                    'V1.norm_input',
    #                    'V1.nonlin_input',
    #                    'V1.conv1',
    #                    'V1.norm1',
    #                    'V1.nonlin1',
    #                    'V1.output',
    #                    'V2.conv_input',
    #                    'V2.norm_input',
    #                    'V2.nonlin_input',
    #                    'V2.conv1',
    #                    'V2.norm1',
    #                    'V2.nonlin1',
    #                    'V2.output',
    #                    'V4.conv_input',
    #                    'V4.norm_input',
    #                    'V4.nonlin_input',
    #                    'V4.conv1',
    #                    'V4.norm1',
    #                    'V4.nonlin1',
    #                    'V4.output',
    #                    'IT.conv_input',
    #                    'IT.norm_input',
    #                    'IT.nonlin_input',
    #                    'IT.conv1',
    #                    'IT.norm1',
    #                    'IT.nonlin1',
    #                    'IT.output',
    #                    'decoder.avgpool',
    #                    'decoder.flatten',
    #                    'decoder.linear'
    #                    ]
    #     comparison_mats_directory = "C:\\Users\\Asus\\Desktop\\Project\\" + model + "\\RDM Comparisons B mats\\"
    #     comparison_mats = []
    #     for u in range(len(type_of_image)):
    #         comparison_mats.append(
    #             scipy.io.loadmat(comparison_mats_directory + 'RDM_comparison_' + type_of_image[u]))
    #     big_values = []
    #     for l in range(len(comparison_mats)):
    #         comparisons = comparison_mats[l]['RDM_comparison' + str(int(l / 3) + 7)]
    #         comparisons = comparisons.squeeze()
    #         data = dict.fromkeys(slice_names, None)
    #         for i in range(len(slice_names)):
    #             data[slice_names[i]] = comparisons[i]
    #         print(comparisons.shape)
    #         slices = list(data.keys())
    #         values = list(data.values())
    #         print(slices)
    #         print(values)
    #         big_values.append(values)
    #     glist = [0, 3]
    #     types = ['Face', 'Body']
    #     oo = 0
    #     for g in glist:
    #         plt.figure(figsize=(20, 20))
    #         plt.yticks(fontsize=20)
    #         plt.ylim(-1.0, 1.0)
    #         plt.legend(fontsize=15)
    #         plt.xticks(fontsize=10, rotation=90)
    #         plt.title(model, fontsize=25)
    #         plt.ylabel("Correlation", fontsize=25)
    #         plt.plot(slices, big_values[g], color=color[g % 3], linewidth=3, label=type_of_image[g])
    #         plt.plot(slices, big_values[g + 1], color=color[(g + 1) % 3], linewidth=3, label=type_of_image[g + 1])
    #         plt.plot(slices, big_values[g + 2], color=color[(g + 2) % 3], linewidth=3, label=type_of_image[g + 2])
    #         plt.legend(fontsize=30)
    #         plt.draw()
    #         plt.savefig(
    #             'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models\\Body and Face plots\\' + model + "_" + types[
    #                 oo] + '.png',
    #
    #             dpi=100)
    #         plt.close()
    #         oo += 1
    #
    # elif model == 'CorNet-Z':
    #     slice_names = ['V1.conv',
    #                    'V1.nonlin',
    #                    'V1.pool',
    #                    'V1.output',
    #
    #                    'V2.conv',
    #                    'V2.nonlin',
    #                    'V2.pool',
    #                    'V2.output',
    #
    #                    'V4.conv',
    #                    'V4.nonlin',
    #                    'V4.pool',
    #                    'V4.output',
    #
    #                    'IT.conv',
    #                    'IT.nonlin',
    #                    'IT.pool',
    #                    'IT.output',
    #
    #                    'decoder.avgpool',
    #                    'decoder.flatten',
    #                    'decoder.linear',
    #                    'decoder.output'
    #                    ]
    #     comparison_mats_directory = "C:\\Users\\Asus\\Desktop\\Project\\" + model + "\\RDM Comparisons B mats\\"
    #     comparison_mats = []
    #     for u in range(len(type_of_image)):
    #         comparison_mats.append(
    #             scipy.io.loadmat(comparison_mats_directory + 'RDM_comparison_' + type_of_image[u]))
    #     big_values = []
    #     for l in range(len(comparison_mats)):
    #         comparisons = comparison_mats[l]['RDM_comparison' + str(int(l / 3) + 7)]
    #         comparisons = comparisons.squeeze()
    #         data = dict.fromkeys(slice_names, None)
    #         for i in range(len(slice_names)):
    #             data[slice_names[i]] = comparisons[i]
    #         print(comparisons.shape)
    #         slices = list(data.keys())
    #         values = list(data.values())
    #         print(slices)
    #         print(values)
    #         big_values.append(values)
    #     glist = [0, 3]
    #     types = ['Face', 'Body']
    #     oo = 0
    #     for g in glist:
    #         plt.figure(figsize=(20, 20))
    #         plt.yticks(fontsize=20)
    #         plt.ylim(-1.0, 1.0)
    #         plt.legend(fontsize=15)
    #         plt.xticks(fontsize=10, rotation=90)
    #         plt.title(model, fontsize=25)
    #         plt.ylabel("Correlation", fontsize=25)
    #         plt.plot(slices, big_values[g], color=color[g % 3], linewidth=3, label=type_of_image[g])
    #         plt.plot(slices, big_values[g + 1], color=color[(g + 1) % 3], linewidth=3, label=type_of_image[g + 1])
    #         plt.plot(slices, big_values[g + 2], color=color[(g + 2) % 3], linewidth=3, label=type_of_image[g + 2])
    #         plt.legend(fontsize=30)
    #         plt.draw()
    #         plt.savefig(
    #             'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models\\Body and Face plots\\' + model + "_" + types[
    #                 oo] + '.png',
    #
    #             dpi=100)
    #         plt.close()
    #         oo += 1

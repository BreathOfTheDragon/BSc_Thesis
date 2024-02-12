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

my_models = ['AlexNet', 'GoogleNet', 'MobileNet-V2', 'ResNet18', 'ResNet50',
             'ResNet101',
             'SqueezeNet1_1',
             'VGG-16', 'VGG-19']
type_of_image = ['Face_HF', 'Face_IF', 'Face_LF', 'Body_HF', 'Body_IF', 'Body_LF']
color = ['black', 'lime', 'orange', 'blue', 'fuchsia', 'red', 'purple', 'grey', 'green', 'cyan', 'saddlebrown']
torchmodels = [models.alexnet(pretrained=True), models.googlenet(pretrained=True), models.mobilenet_v2(pretrained=True),
               models.resnet18(pretrained=True), models.resnet50(pretrained=True), models.resnet101(pretrained=True),
               models.squeezenet1_1(pretrained=True), models.vgg16(pretrained=True), models.vgg19(pretrained=True)]

comparison_mats_directory = []
inx = 0
for model in my_models:
    fig, ax = plt.subplots()
    if model != 'CorNet-S' and model != 'CorNet-R' and model != 'CorNet-Z':
        slice_names = np.array([])  # a list of the names of all slices of the network
        slice_names = np.append(slice_names,
                                np.array(models.feature_extraction.get_graph_node_names(torchmodels[inx])[0]))
        inx += 1
        comparison_mats_directory = "C:\\Users\\Asus\\Desktop\\Project\\" + model + "\\RDM Comparisons B mats\\"
        comparison_mats = []
        fl = [7, 7, 7, 8, 8, 8]
        for u in range(len(type_of_image)):
            comparison_mats.append(
                scipy.io.loadmat(comparison_mats_directory + str(fl[u]) + '_RDM_comparison_' + type_of_image[u]))

        max_comparisons = []
        max_indices = []
        for w in range(len(comparison_mats)):
            comparisons = comparison_mats[w]['RDM_comparison']

            max_comparisons.append(np.amax(comparisons))
            max_indices.append(np.argmax(comparisons))
        p = 0
        for p in range(len(max_indices)):
            max_indices[p] = max_indices[p] + 1
        print(max_comparisons)
        print(max_indices)
        color_list = ['red', 'blue', 'green', 'red', 'blue', 'green']
        plt.figure(figsize=(30, 7))
        # creating the step plot
        plt.ylim(0, 1.0)
        plt.bar(type_of_image, max_comparisons, color=color_list, width=0.2)
        plt.xticks(fontsize=30, rotation=0)
        plt.yticks(fontsize=25)
        plt.ylabel("Correlation", fontsize=25)
        # plt.title("Max correlation between brain region and image types for " + model, fontsize=30)
        plt.draw()
        plt.savefig(
            'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models\\Max Layer plots\\' + model + '_MAX_Layer_HIL_Types.png',
            bbox_inches='tight',
            dpi=100)

        plt.close()
        plt.figure(figsize=(30, 7))
        plt.ylim(0, len(slice_names))
        plt.bar(type_of_image, max_indices, color=color_list, width=0.2)
        plt.xticks(fontsize=30, rotation=0)
        plt.yticks(fontsize=25)
        plt.ylabel("Best Layer", fontsize=25)
        # plt.title("Best CNN layer for each image type , excluding X , for " + model, fontsize=30)
        plt.draw()
        plt.savefig(
            'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models\\Best Layer plots\\' + model + '_Best_Layer_HIL_Types.png',
            bbox_inches='tight',
            dpi=100)

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
    #
    #     for u in range(len(type_of_image)):
    #         comparison_mats.append(
    #             scipy.io.loadmat(comparison_mats_directory + 'RDM_comparison_' + type_of_image[u]))
    #
    #     max_comparisons = []
    #     max_indices = []
    #     for w in range(len(comparison_mats)):
    #         comparisons = comparison_mats[w]['RDM_comparison' + str(int(w / 3) + 7)][1:]
    #
    #         max_comparisons.append(np.amax(comparisons))
    #         max_indices.append(np.argmax(comparisons))
    #     p = 0
    #     for p in range(len(max_indices)):
    #         max_indices[p] = max_indices[p] + 1
    #     print(max_comparisons)
    #     print(max_indices)
    #
    #     plt.figure(figsize=(25, 25))
    #     # creating the step plot
    #     plt.ylim(-1.0, 1.0)
    #     plt.bar(type_of_image, max_comparisons, color='turquoise')
    #     plt.xticks(fontsize=20, rotation=90)
    #     plt.yticks(fontsize=20)
    #     plt.ylabel("Correlation", fontsize=25)
    #     plt.title("Max correlation between brain region and image types for " + model, fontsize=25)
    #     plt.draw()
    #     plt.savefig(
    #         'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models\\Max Layer plots\\' + model + '_MAX_Layer_HIL_Types.png',
    #         dpi=100)
    #
    #     plt.close()
    #     plt.figure(figsize=(25, 25))
    #     plt.ylim(0, len(slice_names))
    #     plt.bar(type_of_image, max_indices, color='red')
    #     plt.xticks(fontsize=20, rotation=90)
    #     plt.yticks(fontsize=20)
    #     plt.ylabel("Best Layer", fontsize=25)
    #     plt.title("Best CNN layer for each image type , excluding X , for " + model, fontsize=25)
    #     plt.draw()
    #     plt.savefig(
    #         'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models\\Best Layer plots\\' + model + '_Best_Layer_HIL_Types.png',
    #         dpi=100)
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
    #
    #     comparison_mats_directory = "C:\\Users\\Asus\\Desktop\\Project\\" + model + "\\RDM Comparisons B mats\\"
    #     comparison_mats = []
    #
    #     for u in range(len(type_of_image)):
    #         comparison_mats.append(
    #             scipy.io.loadmat(comparison_mats_directory + 'RDM_comparison_' + type_of_image[u]))
    #
    #     max_comparisons = []
    #     max_indices = []
    #     for w in range(len(comparison_mats)):
    #         comparisons = comparison_mats[w]['RDM_comparison' + str(int(w / 3) + 7)][1:]
    #
    #         max_comparisons.append(np.amax(comparisons))
    #         max_indices.append(np.argmax(comparisons))
    #     p = 0
    #     for p in range(len(max_indices)):
    #         max_indices[p] = max_indices[p] + 1
    #     print(max_comparisons)
    #     print(max_indices)
    #
    #     plt.figure(figsize=(25, 25))
    #     # creating the step plot
    #     plt.ylim(-1.0, 1.0)
    #     plt.bar(type_of_image, max_comparisons, color='turquoise')
    #     plt.xticks(fontsize=20, rotation=90)
    #     plt.yticks(fontsize=20)
    #     plt.ylabel("Correlation", fontsize=25)
    #     plt.title("Max correlation between brain region and image types for " + model, fontsize=25)
    #     plt.draw()
    #     plt.savefig(
    #         'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models\\Max Layer plots\\' + model + '_MAX_Layer_HIL_Types.png',
    #         dpi=100)
    #
    #     plt.close()
    #     plt.figure(figsize=(25, 25))
    #     plt.ylim(0, len(slice_names))
    #     plt.bar(type_of_image, max_indices, color='red')
    #     plt.xticks(fontsize=20, rotation=90)
    #     plt.yticks(fontsize=20)
    #     plt.ylabel("Best Layer", fontsize=25)
    #     plt.title("Best CNN layer for each image type , excluding X , for " + model, fontsize=25)
    #     plt.draw()
    #     plt.savefig(
    #         'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models\\Best Layer plots\\' + model + '_Best_Layer_HIL_Types.png',
    #         dpi=100)
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
    #
    #     comparison_mats_directory = "C:\\Users\\Asus\\Desktop\\Project\\" + model + "\\RDM Comparisons B mats\\"
    #     comparison_mats = []
    #
    #     for u in range(len(type_of_image)):
    #         comparison_mats.append(
    #             scipy.io.loadmat(comparison_mats_directory + 'RDM_comparison_' + type_of_image[u]))
    #
    #     max_comparisons = []
    #     max_indices = []
    #     for w in range(len(comparison_mats)):
    #         comparisons = comparison_mats[w]['RDM_comparison' + str(int(w / 3) + 7)][1:]
    #
    #         max_comparisons.append(np.amax(comparisons))
    #         max_indices.append(np.argmax(comparisons))
    #     p = 0
    #     for p in range(len(max_indices)):
    #         max_indices[p] = max_indices[p] + 1
    #     print(max_comparisons)
    #     print(max_indices)
    #
    #     plt.figure(figsize=(25, 25))
    #     # creating the step plot
    #     plt.ylim(-1.0, 1.0)
    #     plt.bar(type_of_image, max_comparisons, color='turquoise')
    #     plt.xticks(fontsize=20, rotation=90)
    #     plt.yticks(fontsize=20)
    #     plt.ylabel("Correlation", fontsize=25)
    #     plt.title("Max correlation between brain region and image types for " + model, fontsize=25)
    #     plt.draw()
    #     plt.savefig(
    #         'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models\\Max Layer plots\\' + model + '_MAX_Layer_HIL_Types.png',
    #         dpi=100)
    #
    #     plt.close()
    #     plt.figure(figsize=(25, 25))
    #     plt.ylim(0, len(slice_names))
    #     plt.bar(type_of_image, max_indices, color='red')
    #     plt.xticks(fontsize=20, rotation=90)
    #     plt.yticks(fontsize=20)
    #     plt.ylabel("Best Layer", fontsize=25)
    #     plt.title("Best CNN layer for each image type , excluding X , for " + model, fontsize=25)
    #     plt.draw()
    #     plt.savefig(
    #         'C:\\Users\\Asus\\Desktop\\Project\\Comparing Models\\Best Layer plots\\' + model + '_Best_Layer_HIL_Types.png',
    #         dpi=100)

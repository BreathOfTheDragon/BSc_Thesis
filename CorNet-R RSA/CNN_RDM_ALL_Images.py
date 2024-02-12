import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import rsatoolbox
import pickle
import pandas as pd
import os
import tifffile
from torchvision import models

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

for i in range(len(slice_names)):
    sliceF_x_image = pd.read_pickle(
        r'C:\Users\Asus\Desktop\Project\CorNet-R\CNN Slices Flattened by Image as pickle\\' + str(i+1).zfill(2) + '_' + slice_names[
            i] + '_for_all_images' + '.p')
    sliceF_x_image = sliceF_x_image.transpose()
    slice_data = rsatoolbox.data.Dataset(sliceF_x_image)
    slice_rdm = rsatoolbox.rdm.calc_rdm(slice_data, method='euclidean', descriptor=None, noise=None)
    slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)
    rsatoolbox.vis.show_rdm(slice_rdm_non_square, figsize=(8, 8), show_colorbar='figure')
    plt.draw()
    plt.savefig(
        'C:\\Users\\Asus\\Desktop\\Project\\CorNet-R\\RDMs CNN all images plot\\' + str(i+1).zfill(2) + '_RDM_' + slice_names[i] + '.png',
        dpi=500)
    # rsatoolbox.vis.show
    plt.close()

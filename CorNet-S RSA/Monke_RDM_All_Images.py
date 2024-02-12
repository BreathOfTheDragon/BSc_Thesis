import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import rsatoolbox
import pickle
import pandas as pd

monke_brain_matrix = io.matlab.loadmat(
    'C:\\Users\\Asus\\Desktop\\Project\\neurons stacked\\neurons_stacked_averaged_for_370_to_499.mat')
monke_brain_matrix = monke_brain_matrix['neurons_stacked_averaged_for_370_to_499']
monke_brain_matrix = monke_brain_matrix.transpose()
print(type(monke_brain_matrix))
print(monke_brain_matrix.shape)

monke_data = rsatoolbox.data.Dataset(monke_brain_matrix)
monke_rdm = rsatoolbox.rdm.calc_rdm(monke_data, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)

rsatoolbox.vis.show_rdm(monke_rdm_non_square, figsize=(6, 6), show_colorbar='figure')
# plt.show()
plt.draw()
plt.savefig('C:\\Users\\Asus\\Desktop\\Project\\CorNet-S\\RDMs Monke all images plot\\RDM_Monke_all_images', dpi=500)
# rsatoolbox.vis.show



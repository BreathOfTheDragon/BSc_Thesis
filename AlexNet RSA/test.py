import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import rsatoolbox
import pickle
import scipy
import scipy.spatial
import scipy.stats
import pandas as pd
import os
import tifffile
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

np.random.seed(0)
monke = np.random.rand(6, 352)
slice = np.random.rand(6, 1000)

monkeD = rsatoolbox.data.Dataset(monke)
monke_rdm = rsatoolbox.rdm.calc_rdm(monkeD, method='euclidean', descriptor=None, noise=None)
monke_rdm_non_square = rsatoolbox.rdm.sqrt_transform(monke_rdm)

sliceD = rsatoolbox.data.Dataset(slice)
slice_rdm = rsatoolbox.rdm.calc_rdm(sliceD, method='euclidean', descriptor=None, noise=None)
slice_rdm_non_square = rsatoolbox.rdm.sqrt_transform(slice_rdm)

print(rsatoolbox.rdm.compare_rho_a(monke_rdm_non_square, slice_rdm_non_square))
print(rsatoolbox.rdm.compare_rho_a(monke_rdm, slice_rdm))

monke_distances = scipy.spatial.distance.pdist(monke, metric='euclidean')
print(monke_rdm)
print(monke_distances)
slice_distances = scipy.spatial.distance.pdist(slice, metric='euclidean')
print(scipy.stats.pearsonr(monke_distances, slice_distances))

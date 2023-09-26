import numpy as np
import pandas as pd
import scanpy as sc
import scvelo  ## For mouse gastrulation data 
import anndata
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
plt.rcParams['figure.figsize']=(8,8) #rescale figures
sc.settings.verbosity = 3

import milopy.core as milo
import milopy.plot as milopl

#adata = sc.read('/root/data/totalVI_cite_pbmc_integrated.h5ad')

adata_original = anndata.read_h5ad("/root/scarches_mapped_meyer.h5ad")

adata = sc.pp.subsample(adata_original, fraction=0.01, n_obs=900, random_state=0, copy=True)

adata_condition = adata[adata.obs['dataset'] == "Barbry_Leroy_2020"].copy()
adata_control = adata[adata.obs['dataset'] != "Barbry_Leroy_2020"].copy()

sc.tl.pca(adata_control)
sc.tl.pca(adata_condition)

d = 30
k = 50

print(len(adata_control.obsm["X_pca"][0]) )
print(adata_control.X[0]) 

adata_condition.obs["X_probabilities"] = [0] * len(adata_condition.X)
data = np.zeros(len(adata_condition.X))

for i in range(0, len(adata_condition.X)):
    cell_condition = adata_condition.X[i]
    distances = list()
    for cell_control in adata_control.X:
        cell_control_sqrt = 0
        for i1 in range(0, len(cell_control)):
            cell_control_sqrt += (cell_condition[i1] - cell_control[i1])**2
        cell_control_sqrt = cell_control_sqrt ** 0.5
        distances.append(cell_control_sqrt)
    avg_distance = sum(sorted(distances)[:10]) / 10
    adata_condition.obs["X_probabilities"][i] = avg_distance
    data[i] = avg_distance

input_dim = 1  # Dimensionality of the input data

# Encoder
encoder = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu')
])

# Decoder
decoder = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(input_dim, activation='linear')
])

# Autoencoder
autoencoder = keras.Sequential([encoder, decoder])

autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(adata_condition.obs["X_probabilities"], adata_condition.obs["X_probabilities"], epochs=50, batch_size=32, shuffle=True)

reconstructed_data = autoencoder.predict(data)
print(adata_condition.obs["X_probabilities"])
print(reconstructed_data)
mse = np.mean((data - reconstructed_data), axis=1)  # Calculate MSE for OOD data

max = np.max(data)
average = np.average(data)
for i in range(0, len(adata_condition.obs["X_probabilities"])):
    adata_condition.obs["X_probabilities"][i] = (adata_condition.obs["X_probabilities"][i] - average)/(max - average)#bad but easy way to calculate probabilities

#plt.hist(adata_condition.obs["X_probabilities"], bins=100)
plt.hist(data-reconstructed_data, bins=100)

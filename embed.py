import torch
from torch import nn
import numpy as np
import sys
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import importlib

sys.path.append("..")
from model import ParticleEventTransformer
from data import get_database_path, get_h5_files, read_h5_file, select_events
from utils import load_toml_config

import matplotlib.pyplot as plt
from analysis import inference
from data import EventDataset
main_folder = os.path.dirname(__file__)


device = "cuda" if torch.cuda.is_available() else "mps" if sys.platform == "darwin" else "cpu"
random_seed = 114514
torch.manual_seed(random_seed)
np.random.seed(random_seed)

EMD_config = load_toml_config("EMD")
particle_type_scale = EMD_config['particle_type_scale']

# Load raw events
files = load_toml_config("file")
database_path = get_database_path()

events = {}
for key, value in files.items():
    events[key] = read_h5_file(database_path, value)

Delphes = np.load(os.path.join(main_folder, 'data', 'datasets_-1.npz'))
unpreprocessed_labels = ['x_train','x_test', 'x_val']
full_SM_dataset = np.concatenate([Delphes[label] for label in unpreprocessed_labels], axis=0)
events['SM'] = full_SM_dataset.reshape(full_SM_dataset.shape[:3])


signals = [key for key in events.keys() if key != "SM"]
print(signals)

# Load model

model_hyper_parameters = load_toml_config("Transformer")
print(model_hyper_parameters)
feature_size = model_hyper_parameters["feature_size"]
embed_size = model_hyper_parameters["embed_size"]
num_heads = model_hyper_parameters["num_heads"]
num_layers = model_hyper_parameters["num_layers"]
hidden_dim = model_hyper_parameters["hidden_dim"]
output_dim = model_hyper_parameters["output_dim"]

embedding_model = ParticleEventTransformer(feature_size, embed_size, num_heads, hidden_dim, output_dim, num_layers)
model_name = "emb_dim{}_type_scale{}.pt".format(output_dim, particle_type_scale) if EMD_config['pid_method'] == 'one-hot' else "emb_dim{}_sep.pt".format(output_dim)
embedding_model.load_state_dict(torch.load(os.path.join(main_folder, "model", model_name)))
embedding_model.to(device)

infer_test_num = 1000000
dataloaders = {}
# for key, value in events.items():
#     dataloaders[key] = DataLoader(EventDataset(value[:infer_test_num]), batch_size=256, num_workers=16, prefetch_factor=5)

for key, value in events.items():
    dataloaders[key] = DataLoader(EventDataset(value), batch_size=256, num_workers=16, prefetch_factor=5)

embedding_points = {}

for key, value in dataloaders.items():
    embedding_points[key] = inference(embedding_model, value, embed_dim=output_dim)
    print(key, embedding_points[key].shape)

# Save embedding points in HDF5 file
import h5py

embedding_points_file_name = "embedding_points_dim{}.h5".format(output_dim)

embedding_points_file = os.path.join(main_folder, 'latent', embedding_points_file_name)
with h5py.File(embedding_points_file, "w") as f:
    for key, value in embedding_points.items():
        f.create_dataset(key, data=value)
print("Embedding points saved as:", embedding_points_file)
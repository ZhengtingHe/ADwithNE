import json
import sys
import h5py
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from emd import emd_pot, sep_emd
from concurrent.futures import ProcessPoolExecutor
from utils import load_toml_config



def load_config():
    with open('../config.json', 'r') as f:
        config = json.load(f)
    return config


def get_database_path():
    config = load_toml_config('data_path')
    if sys.platform.startswith('linux'):
        return config['linux']
    elif sys.platform.startswith('win32') or sys.platform.startswith('cygwin'):
        return config['windows']
    else:
        raise Exception("Unsupported OS")


def get_h5_files():
    config = load_config()
    return config['bkg_files'], config['sig_files']


def read_h5_file(path, file_name, datatype='Particles'):
    if datatype == 'Particles':
        events = np.array(h5py.File(os.path.join(path, file_name), 'r')[datatype])
        # Rotate the events so that MET is at phi = 0
        events[:,:,2] = (events[:,:,2] - events[:,0,2].reshape(len(events), 1)) % np.pi 
        return events
    elif datatype == "EMD":
        data = h5py.File(os.path.join(path, file_name), "r")
        return np.array(data["pairs"]), np.array(data["emds"])
    

def select_events(events, criteria):
    # crtieria [int, int, int] for number of [electron, muon, jets]
    count_elctron = np.sum(events[:,:,3] == 2, axis=1)
    count_muon = np.sum(events[:,:,3] == 3, axis=1)
    count_jet = np.sum(events[:,:,3] == 4, axis=1)

    mask = (count_elctron == criteria[0]) & (count_muon == criteria[1]) & (count_jet == criteria[2])
    return events[mask]


def sample_pairs(n_events, n_pairs):
    # np.random.choice can be extremely slow, use randint instead
    pairs = np.random.randint(0, n_events, (n_pairs, 2))
    # remove pairs with same index since we don't want to compare the same event
    bad_samples_idx = np.where(pairs[:, 0] == pairs[:, 1])
    for i in bad_samples_idx:
        pairs[i] = np.random.choice(n_events, 2, replace=False)
    return pairs


def sample_matrix(n_events, pairs):
    matrix = np.zeros((n_events, n_events))
    for pair in pairs:
        matrix[pair[0], pair[1]] += 1
    return matrix


def sample_pairs_with_emd(events, n_pairs=None, particle_type_scale=0, norm=False, pid_method='one-hot'):
    n_events = len(events)
    if n_pairs is None:
        n_pairs = 5 * n_events
    pairs = sample_pairs(n_events, n_pairs)
    emds = np.zeros(n_pairs)
    if pid_method == 'one-hot':
        for i, pair in enumerate(tqdm(pairs)):
            emds[i] = emd_pot(events[pair[0]], events[pair[1]], particle_type_scale=particle_type_scale, norm=norm, particle_one_hot=True)
    elif pid_method == 'separate':
        for i, pair in enumerate(tqdm(pairs)):
            emds[i] = sep_emd(events[pair[0]], events[pair[1]])
    return pairs, emds



def sample_pairs_with_emd_multi(events, n_pairs=None, particle_type_scale=0, norm=False, pid_method='one-hot', n_jobs=12):
    def compute_emd_one_hot(pair, events, particle_type_scale, norm):
        return emd_pot(events[pair[0]], events[pair[1]], particle_type_scale=particle_type_scale, norm=norm, particle_one_hot=True)

    def compute_emd_separate(pair, events):
        return sep_emd(events[pair[0]], events[pair[1]])
    n_events = len(events)
    if n_pairs is None:
        n_pairs = 5 * n_events
    pairs = sample_pairs(n_events, n_pairs)
    emds = np.zeros(n_pairs)

    if pid_method == 'one-hot':
        compute_emd = compute_emd_one_hot
        compute_emd_args = (events, particle_type_scale, norm)
    elif pid_method == 'separate':
        compute_emd = compute_emd_separate
        compute_emd_args = (events,)
    else:
        raise ValueError(f"Unsupported pid_method: {pid_method}")

    def compute_emd_wrapper(pair):
        return compute_emd(pair, *compute_emd_args)

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(tqdm(executor.map(compute_emd_wrapper, pairs), total=n_pairs))
    
    emds = np.array(results)
    return pairs, emds

def store_emds_with_pairs(emds, pairs, file_name):
    f = h5py.File(os.path.join("..","generated_data", file_name), "w")
    f["pairs"] = pairs
    f["emds"] = emds
    f.close()

def sample_pairs_from_diff_dataset(events1, events2, n_pairs, particle_type_scale=0, norm=False, pid_method='one_hot'):
    index1 = np.random.randint(0, len(events1), n_pairs)
    index2 = np.random.randint(0, len(events2), n_pairs)
    pairs = np.stack([index1, index2], axis=1)
    emds = np.zeros(n_pairs)
    if pid_method == 'one_hot':
        for i, (idx1, idx2) in enumerate(tqdm(pairs)):
            emds[i] = emd_pot(events1[idx1], events2[idx2], particle_type_scale=particle_type_scale, norm=norm)
    elif pid_method == 'separate':
        for i, (idx1, idx2) in enumerate(tqdm(pairs)):
            emds[i] = sep_emd(events1[idx1], events2[idx2])
    return pairs, emds

class PairedEventsDataset(Dataset):
    def __init__(self, events, pairs, emds):
        assert len(emds) == len(pairs)
        assert len(pairs.shape) == 2 and pairs.shape[1] == 2
        self.events = torch.from_numpy(events[:, :, :3])
        self.pairs = torch.from_numpy(pairs)
        self.emds = torch.from_numpy(emds)
        
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.events[self.pairs[idx][0]], self.events[self.pairs[idx][1]], self.emds[idx]
    

class RealTimeEMDDataset(Dataset):
    def __init__(self, events, EMD_fn=emd_pot, n_pairs=100000):
        self.events = events
        self.tensor_events = torch.from_numpy(events[:, :, :3])
        self.n_events = len(events)
        self.n_pairs = n_pairs
        self.EMD_fn = EMD_fn
    
    def __len__(self):
        return self.n_pairs
    
    def __getitem__(self, idx):
        pair = sample_pairs(self.n_events, 1)[0]
        source_event = self.events[pair[0]]
        target_event = self.events[pair[1]]
        return self.tensor_events[pair[0]], self.tensor_events[pair[1]], torch.Tensor(self.EMD_fn(source_event, target_event))


class EventDataset(Dataset):
    def __init__(self, events):
        self.events = torch.from_numpy(events[:, :, :3],).float()
    
    def __len__(self):
        return len(self.events)
    
    def __getitem__(self, idx):
        return self.events[idx]
    


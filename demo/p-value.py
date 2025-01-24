import torch
from torch import nn
import numpy as np
import sys
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import importlib
import h5py

sys.path.append("..")
from model import ParticleEventTransformer
from data import get_database_path, get_h5_files, read_h5_file, select_events
from utils import load_toml_config
from analysis import Normalizer
from classify import train_model

import matplotlib.pyplot as plt

import importlib
import classify
import analysis
importlib.reload(classify)
importlib.reload(analysis)
from classify import train_model
from analysis import Bootstrap_Permutation
opt_train_model  = torch.compile(train_model)
from analysis import LambdaEstimator
from metrics import BinaryACCUpdater
from analysis import create_exp_bkg_events, train_test_split, get_dataloaders
from model import MLP

device = "cuda" if torch.cuda.is_available() else "mps" if sys.platform == "darwin" else "cpu"
random_seed = 114514
torch.manual_seed(random_seed)
np.random.seed(random_seed)

database_path = "/home/desmondhe/ADwithNE/latent"
output_dim = [2,4,8,16,32]
embedding_points_file_name = {}
embedding_points_file = {}
embedding_points = {}
targets = {}
for dim in output_dim:
    embedding_points_file_name[dim] = "embedding_points_dim{}.h5".format(dim)
    embedding_points_file[dim] = os.path.join(database_path, embedding_points_file_name[dim])
    embedding_points[dim] = h5py.File(embedding_points_file[dim])
bsm_events = ['charged_Higgs', 'leptoquark', 'neutral_Higgs', 'neutral_boson']
# Load bkg labels
bkg_id = np.load('../data/background_IDs_-1.npz')
set_labels = ['background_ID_train', 'background_ID_test', 'background_ID_val']
bkg_labels = np.concatenate([bkg_id[label] for label in set_labels], axis=0)

assert embedding_points[2]['SM'].shape[0] == len(bkg_labels)

n_bootstrap = 5


test_ratio=0.2
val_ratio = 0.2
n = 200000
n_null = 1000

# test_signal = "neutral_boson"
test_signal = "leptoquark"
# test_signal = "neutral_Higgs"
# test_signal = "charged_Higgs"
significance = 0.05

database_path = "/home/desmondhe/ADwithNE/latent"
output_dim = [2,4,8,16,32]
# output_dim = [2,4]
test_lambdas = [0.005, 0.01, 0.02, 0.05, 0.1]

lrt_permuation_p_value_mean = np.zeros((len(output_dim), len(test_lambdas)))
lrt_permuation_p_value_std = np.zeros((len(output_dim), len(test_lambdas)))
lrt_bootstrap_p_value_mean = np.zeros((len(output_dim), len(test_lambdas)))
lrt_bootstrap_p_value_std = np.zeros((len(output_dim), len(test_lambdas)))
                                     
auc_permuation_p_value_mean = np.zeros((len(output_dim), len(test_lambdas)))
auc_permuation_p_value_std = np.zeros((len(output_dim), len(test_lambdas)))
auc_bootstrap_p_value_mean = np.zeros((len(output_dim), len(test_lambdas)))
auc_bootstrap_p_value_std = np.zeros((len(output_dim), len(test_lambdas)))
                                     
mce_permuation_p_value_mean = np.zeros((len(output_dim), len(test_lambdas)))
mce_permuation_p_value_std = np.zeros((len(output_dim), len(test_lambdas)))
mce_bootstrap_p_value_mean = np.zeros((len(output_dim), len(test_lambdas)))
mce_bootstrap_p_value_std = np.zeros((len(output_dim), len(test_lambdas)))

lambda_mean = np.zeros((len(output_dim), len(test_lambdas)))
lambda_std = np.zeros((len(output_dim), len(test_lambdas)))
for a, test_dimension in enumerate(output_dim):
    normalizer = Normalizer(*[value for value in embedding_points[test_dimension].values()])
    for i in tqdm(range(len(test_lambdas)), position=0, leave=True):
        sig_lambda = test_lambdas[i]

        estimated_lambdas = np.zeros(n_bootstrap)
        lrt_permutation_p_values = np.zeros(n_bootstrap)
        lrt_bootstrap_p_values = np.zeros(n_bootstrap)
        auc_permutation_p_values = np.zeros(n_bootstrap)
        auc_bootstrap_p_values = np.zeros(n_bootstrap)
        mce_permutation_p_values = np.zeros(n_bootstrap)
        mce_bootstrap_p_values = np.zeros(n_bootstrap)
        
        for j in tqdm(range(n_bootstrap), position=1, leave=False):
            exp_events, bkg_events = create_exp_bkg_events(np.array(embedding_points[test_dimension]['SM']), np.array(embedding_points[test_dimension][test_signal]), sig_lambda, n)
            X1, X2, W1, W2 = train_test_split(exp_events, bkg_events, test_ratio)
            n1 = len(W1)
            m1 = len(X1)
            n2 = len(W2)
            m2 = len(X2)
            pi = n1 / (n1 + m1)
            train_dataloader, val_dataloader = get_dataloaders(X1, W1, val_ratio, normalizer)

            hidden_dim = [8, 16, 16, 16, 8]
            naive_model = MLP(test_dimension, hidden_sizes=hidden_dim)
            naive_model.to(device)
            optimizer = torch.optim.Adam(naive_model.parameters(), lr=1e-3, weight_decay=1e-5)
            loss_fn = nn.BCELoss()
            acc_metric = BinaryACCUpdater()
            metric_dict = {"Accuracy": acc_metric}

            train_model(
                naive_model, optimizer,
                loss_fn, metrics_dict=metric_dict,
                train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                monitor="val_Accuracy", mode="max",
                epochs=50,
                verbose=False
                )
            
            bootstrap_permutation = Bootstrap_Permutation(X2, W2, naive_model, pi, normalizer)
            _, _, _ = bootstrap_permutation.bootstrap(n_null, verbose=False, n_jobs=48)
            _, _, _ = bootstrap_permutation.permutation(n_null, verbose=False, n_jobs=48)

            lrt_permutation_p_values[j] = bootstrap_permutation.lrt_p_permutation
            lrt_bootstrap_p_values[j] = bootstrap_permutation.lrt_p_bootstrap
            auc_permutation_p_values[j] = bootstrap_permutation.auc_p_permutation
            auc_bootstrap_p_values[j] = bootstrap_permutation.auc_p_bootstrap
            mce_permutation_p_values[j] = bootstrap_permutation.mce_p_permutation
            mce_bootstrap_p_values[j] = bootstrap_permutation.mce_p_bootstrap



            # lambda_estimator = LambdaEstimator(X2, W2, naive_model, T=0.5, n_bins=20, normalizer=normalizer)
            # estimated_lambdas[j] = lambda_estimator.estimated_lambda
        # print("sig_lambda:", sig_lambda, "estimated lambda mean:", estimated_lambdas.mean(), "estimated lambda std:", estimated_lambdas.std())
        # lambda_mean[a][i] = estimated_lambdas.mean()
        # lambda_std[a][i] = estimated_lambdas.std(ddof = 1)

        lrt_permuation_p_value_mean[a][i] = np.mean(lrt_permutation_p_values)
        lrt_permuation_p_value_std[a][i] = np.var(lrt_permutation_p_values, ddof=1)
        lrt_bootstrap_p_value_mean[a][i] = np.mean(lrt_bootstrap_p_values)
        lrt_bootstrap_p_value_std[a][i] = np.var(lrt_bootstrap_p_values, ddof=1)

        auc_permuation_p_value_mean[a][i] = np.mean(auc_permutation_p_values)
        auc_permuation_p_value_std[a][i] = np.var(auc_permutation_p_values, ddof=1)
        auc_bootstrap_p_value_mean[a][i] = np.mean(auc_bootstrap_p_values)
        auc_bootstrap_p_value_std[a][i] = np.var(auc_bootstrap_p_values, ddof=1)

        mce_permuation_p_value_mean[a][i] = np.mean(mce_permutation_p_values)
        mce_permuation_p_value_std[a][i] = np.var(mce_permutation_p_values, ddof=1)
        mce_bootstrap_p_value_mean[a][i] = np.mean(mce_bootstrap_p_values)
        mce_bootstrap_p_value_std[a][i] = np.var(mce_bootstrap_p_values, ddof=1)

np.savez('../paper/Kuusela'+test_signal+'.npz', 
        lrt_bootstrap_p_value_mean=lrt_bootstrap_p_value_mean,
        lrt_bootstrap_p_value_std=lrt_bootstrap_p_value_std,
        lrt_permuation_p_value_mean=lrt_permuation_p_value_mean,
        lrt_permuation_p_value_std=lrt_permuation_p_value_std,
        auc_bootstrap_p_value_mean=auc_bootstrap_p_value_mean,
        auc_bootstrap_p_value_std=auc_bootstrap_p_value_std,
        auc_permuation_p_value_mean=auc_permuation_p_value_mean,
        auc_permuation_p_value_std=auc_permuation_p_value_std,
        mce_bootstrap_p_value_mean=mce_bootstrap_p_value_mean,
        mce_bootstrap_p_value_std=mce_bootstrap_p_value_std,
        mce_permuation_p_value_mean=mce_permuation_p_value_mean,
        mce_permuation_p_value_std=mce_permuation_p_value_std,
        lambda_mean=lambda_mean,
        lambda_std=lambda_std
        )

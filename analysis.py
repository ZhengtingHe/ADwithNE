import torch
import numpy as np
from tqdm import tqdm
import sys
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from scipy.optimize import curve_fit
from utils import load_toml_config

sys.path.append("..")

# device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "mps" if sys.platform == "darwin" else "cpu"

@torch.no_grad()
def inference(model, dataloader, embed_dim=2):
    model.eval()
    batch_size = dataloader.batch_size
    embed = np.zeros((len(dataloader.dataset), embed_dim))
    for i, event in enumerate(tqdm(dataloader)):
        event = event.to(device)
        output = model(event)
        embed[i * batch_size:(i + 1) * batch_size] = output.cpu().numpy()
    return embed

def create_exp_bkg_events(ori_bkg_events, ori_sig_events, sig_lambda, n=100000):
    """
    Create expperiment events by combining pure signal and background events with signal strength lambda, 
    and return the background events from remaining pure background events.
    Background: X = {X1,...,X_mb}, Xi ∼ pb Signal: Y = {Y1,...,Y_ms}, Experimental: W = {W1,...,W_n},
    """
    assert n * sig_lambda <= len(ori_sig_events)
    m_s = int(n * sig_lambda)
    m_b = n - m_s
    choosen_bkg_events = ori_bkg_events[np.random.choice(len(ori_bkg_events), m_b + n, replace=False)]
    choosen_sig_events = ori_sig_events[np.random.choice(len(ori_sig_events), m_s, replace=False)]
    exp_events = np.concatenate((choosen_bkg_events[:m_b], choosen_sig_events))
    np.random.shuffle(exp_events)
    bkg_events = choosen_bkg_events[m_b:m_b + n]
    np.random.shuffle(bkg_events)
    return exp_events, bkg_events

def create_exp_bkg_from_multi_sig(events, sig_formula, n=100000):
    assert type(sig_formula) == dict
    for key, value in sig_formula.items():
        assert value * n <= len(events[key])
    bkg_ratio = 1 - sum(sig_formula.values())
    m_b = int(n * bkg_ratio)
    exp_events = np.concatenate([events[key][:int(n * value)] for key, value in sig_formula.items()] + [events["bkg"][:m_b]])
    np.random.shuffle(exp_events)
    bkg_events = events["SM"][m_b:m_b + n]
    np.random.shuffle(bkg_events)
    return exp_events, bkg_events

def create_exp_null(bkg_events, n=100000):
    """
    Create pure experiment events for estimation of null distribution
    """
    choosen_events = bkg_events[np.random.choice(len(bkg_events), 2 * n, replace=False)]
    exp_events = choosen_events[:n]
    np.random.shuffle(exp_events)
    bkg_events = choosen_events[n:]
    np.random.shuffle(bkg_events)
    return exp_events, bkg_events

def train_test_split(exp_events, bkg_events, test_ratio=0.2):
    """
    Split background data X = {X1,...,X_mb} into X1 and X2 of sizes m1 and m2 respectively.
    Split experimental data W = {W1,...,W_n} into W1 and W2 of sizes n1 and n2 respectively, with n2 = m2.
    Will assume n = mb for now.
    """
    n1 = int((1 - test_ratio) * len(exp_events))
    m1 = n1
    np.random.shuffle(exp_events)
    np.random.shuffle(bkg_events)
    X1 = bkg_events[:m1]
    X2 = bkg_events[m1:]
    W1 = exp_events[:n1]
    W2 = exp_events[n1:]
    return X1, X2, W1, W2

class Normalizer:
    def __init__(self, *points):
        self.max = np.max(np.concatenate(points), axis=0)
        self.min = np.min(np.concatenate(points), axis=0)

    def __call__(self, x):
        return (x - self.min) / (self.max - self.min)

class ClassifyDataset(Dataset):
    def __init__(self, exp_events, bkg_events, normalizer):
        """
        Experiment events labeled as 1, background events labeled as 0.
        """
        # Normalize the data
        if normalizer is not None:
            exp_events = normalizer(exp_events)
            bkg_events = normalizer(bkg_events)

        bkg_events = torch.from_numpy(bkg_events).float()
        exp_events = torch.from_numpy(exp_events).float()
        self.events = torch.cat([bkg_events, exp_events], dim=0)
        self.labels = torch.cat([torch.zeros(len(bkg_events)), torch.ones(len(exp_events))])
    
    def __len__(self):
        return len(self.events)
    
    def __getitem__(self, idx):
        return self.events[idx], self.labels[idx]
    

def get_dataloaders(X1, W1, val_ratio, normalizer):
    """
    Get dataloaders for training and validation sets.
    """
    n_train = int((1 - val_ratio) * len(W1))
    X1_train = X1[:n_train]
    X1_val = X1[n_train:]
    W1_train = W1[:n_train]
    W1_val = W1[n_train:]
    
    train_dataset = ClassifyDataset(W1_train, X1_train, normalizer=normalizer)
    val_dataset = ClassifyDataset(W1_val, X1_val, normalizer=normalizer)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=64)
    val_dataloader = DataLoader(val_dataset, batch_size=256)
    return train_dataloader, val_dataloader

@torch.no_grad()
def predict(model, dataloader):
    model.eval()
    batch_size = dataloader.batch_size
    targets = np.zeros(len(dataloader.dataset))
    predictions = np.zeros(len(dataloader.dataset))
    for i, (features, labels) in enumerate(tqdm(dataloader)):
        features = features.to(device)
        outputs = model(features)
        targets[i * batch_size: (i + 1) * batch_size] = labels.numpy()
        predictions[i * batch_size: (i + 1) * batch_size] = outputs.cpu().numpy()
    return targets, predictions

def calculate_auc(targets, predictions):
    fpr, tpr, _ = metrics.roc_curve(targets, predictions)
    auc_roc = metrics.auc(fpr, tpr)
    return fpr, tpr, auc_roc

class TestDataset(Dataset):
    def __init__(self, events, normalizer):
        if normalizer:
            events = normalizer(events)
        self.events = torch.from_numpy(events).float()
    
    def __len__(self):
        return len(self.events)
    
    def __getitem__(self, idx):
        return self.events[idx]
    
@torch.no_grad()
def h_hat(classifier, W, normalizer=None):
    classifier.eval()
    W_dataset = TestDataset(W, normalizer=normalizer)
    batch_size = 512
    W_dataloader = DataLoader(W_dataset, batch_size=batch_size)
    predictions = np.zeros(len(W))
    for i, features in enumerate(W_dataloader):
        features = features.to(device)
        outputs = classifier(features)
        predictions[i * batch_size: (i + 1) * batch_size] = outputs.cpu().numpy()
    return predictions

def lrt(h_W, pi):
    n2 = len(h_W)
    return np.log((1 - pi) / pi) + (1 / n2) * np.sum(np.log(h_W / (1 - h_W)))


def auc(h_W, h_X):
    m2 = len(h_X)
    n2 = len(h_W)
    result = h_W[:, None] > h_X[None, :]
    sum = np.sum(result)
    return sum / (m2 * n2)
    

def mce(h_W, h_X, pi):
    m2 = len(h_X)
    n2 = len(h_W)
    x_sum = np.sum(h_X > pi)
    w_sum = np.sum(h_W < pi)
    return 0.5 * ((1/m2) * x_sum + (1/n2) * w_sum)

from concurrent.futures import ProcessPoolExecutor

class Bootstrap_Permutation:
    def __init__(self, X2, W2, classifier, pi, normalizer):
        assert len(X2) == len(W2) # Make sure n2= m2
        self.m2 = len(X2)
        self.n2 = len(W2)
        self.n_union = self.m2 + self.n2
        self.pi = pi

        self.X2 = X2
        self.W2 = W2
        self.union = np.concatenate((self.X2, self.W2))

        self.h_X2 = h_hat(classifier, self.X2, normalizer=normalizer)
        self.h_W2 = h_hat(classifier, self.W2, normalizer=normalizer)
        self.h_union = h_hat(classifier, self.union, normalizer=normalizer)

        self.lrt_exp = lrt(self.h_W2, pi)
        self.auc_exp = auc(self.h_W2, self.h_X2)
        self.mce_exp = mce(self.h_W2, self.h_X2, pi)

    def _bootstrap_iteration(self, seed):
        rng = np.random.RandomState(seed)
        lrt_val = lrt(self.h_union[rng.randint(0, self.n_union, self.n2)], self.pi)
        auc_val = auc(self.h_union[rng.randint(0, self.n_union, self.n2)], self.h_union[rng.randint(0, self.n_union, self.m2)])
        mce_val = mce(self.h_union[rng.randint(0, self.n_union, self.n2)], self.h_union[rng.randint(0, self.n_union, self.m2)], self.pi)
        return lrt_val, auc_val, mce_val

    def bootstrap(self, n, verbose=True, n_jobs=24):
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(tqdm(executor.map(self._bootstrap_iteration, range(n), chunksize=50), total=n, disable=not verbose))

        lrt_null, auc_null, mce_null = zip(*results)

        # P-value
        lrt_p = np.mean(lrt_null > self.lrt_exp) if self.lrt_exp > np.mean(lrt_null) else np.mean(lrt_null < self.lrt_exp)
        auc_p = np.mean(auc_null > self.auc_exp) if self.auc_exp > np.mean(auc_null) else np.mean(auc_null < self.auc_exp)
        mce_p = np.mean(mce_null > self.mce_exp) if self.mce_exp > np.mean(mce_null) else np.mean(mce_null < self.mce_exp)

        self.lrt_p_bootstrap = lrt_p
        self.auc_p_bootstrap = auc_p
        self.mce_p_bootstrap = mce_p

        return lrt_null, auc_null, mce_null

    def _permutation_iteration(self, seed):
        rng = np.random.RandomState(seed)
        sample1 = self.h_union.copy()
        sample2 = self.h_union.copy()
        rng.shuffle(sample1)
        rng.shuffle(sample2)
        lrt_val = lrt(self.h_union[rng.choice(self.n_union, self.n2, replace=False)], self.pi)
        auc_val = auc(sample1[:self.n2], sample1[self.n2:])
        mce_val = mce(sample2[:self.n2], sample2[self.n2:], self.pi)
        return lrt_val, auc_val, mce_val

    def permutation(self, n, verbose=True, n_jobs=24):
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(tqdm(executor.map(self._permutation_iteration, range(n), chunksize=50), total=n, disable=not verbose))

        lrt_null, auc_null, mce_null = zip(*results)

        # P-value
        lrt_p = np.mean(lrt_null > self.lrt_exp) if self.lrt_exp > np.mean(lrt_null) else np.mean(lrt_null < self.lrt_exp)
        auc_p = np.mean(auc_null > self.auc_exp) if self.auc_exp > np.mean(auc_null) else np.mean(auc_null < self.auc_exp)
        mce_p = np.mean(mce_null > self.mce_exp) if self.mce_exp > np.mean(mce_null) else np.mean(mce_null < self.mce_exp)

        self.lrt_p_permutation = lrt_p
        self.auc_p_permutation = auc_p
        self.mce_p_permutation = mce_p

        return lrt_null, auc_null, mce_null
    
lambda_est_params = load_toml_config("lambda_estimate")
class LambdaEstimator:
    def __init__(self, X2, W2, classifier, normalizer, T=lambda_est_params['T'], n_bins=lambda_est_params['n_bins']):
        assert len(X2) == len(W2) # Make sure n2= m2
        self.m2 = len(X2)
        self.n2 = len(W2)

        self.X2 = X2
        self.W2 = W2

        self.h_X2 = h_hat(classifier, self.X2, normalizer=normalizer)
        self.h_W2 = h_hat(classifier, self.W2, normalizer=normalizer)
        result = self.h_X2[:, None] > self.h_W2[None, :]
        self.rho_W = np.sum(result, axis=0) / self.m2
        self.T = T
        self.bins = np.linspace(0, 1, n_bins+1)
        self.H_t = np.histogram(self.rho_W, bins=self.bins, density=True)[0]
        assert T in self.bins
        self.fit_start_idx = np.where(self.bins == T)[0][0]
        # fit a Poisson regression f(t) = exp(β0+ β1t)
        def poisson(t, beta0, beta1):
            return np.exp(beta0 + beta1 * t)
        def exp_const(_, beta0):
            return np.exp(beta0)
        opt, _ = curve_fit(poisson, self.bins[self.fit_start_idx:], self.H_t[self.fit_start_idx-1:], p0=[0, 0])
        self.beta0, self.beta1 = opt
        # Constrain β1 ≤ 0, refit β0 when β1 > 0
        if self.beta1 > 0:
            self.beta1 = 0
            opt, _ = curve_fit(exp_const, self.bins[self.fit_start_idx:], self.H_t[self.fit_start_idx-1:], p0=[0])
            self.beta0 = opt[0]
        self.estimated_H_t = poisson(self.bins[self.fit_start_idx:], self.beta0, self.beta1)
        self.estimated_lambda = 1 - poisson(1, self.beta0, self.beta1)


import sys
import torch
from torch import nn, Tensor
from torchmetrics.regression import MeanAbsolutePercentageError



device = "cuda" if torch.cuda.is_available() else "mps" if sys.platform == "darwin" else "cpu"
MAPE = MeanAbsolutePercentageError().to(device)

def Euclidean_distance(y1, y2):
    return torch.norm(y1 - y2, dim=1)

def hyperbolic_distance(y1, y2):
    return torch.acosh(1 + 2 * torch.sum((y1 - y2)**2, dim=1) / ((1 - torch.sum(y1**2, dim=1)) * (1 - torch.sum(y2**2, dim=1))))

def MAPE_dispersion(original_distance, embed_distance):
        dispersion_emd = torch.var(original_distance, dim=0) / torch.mean(original_distance, dim=0)
        dispersion_ori = torch.var(embed_distance, dim=0) / torch.mean(embed_distance, dim=0)
        return MAPE(dispersion_emd, dispersion_ori)

def embed_ratio(embed_distance, original_distance):
    return torch.mean(embed_distance / original_distance)

class MetricUpdater(nn.Module):
    def __init__(self, metric_fn):
        super(MetricUpdater, self).__init__()
        self.metric_fn = metric_fn
        self.total_metric = 0
        self.count = 0
    def forward(self, original, embed):
        metric = self.metric_fn(embed, original)
        self.update(metric)
        return metric
    def update(self, metric):
        self.total_metric += metric
        self.count += 1
    def compute(self):
        return self.total_metric / self.count
    def reset(self):
        self.total_metric = 0
        self.count = 0

class BinaryACCUpdater(nn.Module):
    def __init__(self, threshold=0.5):
        super(BinaryACCUpdater, self).__init__()
        self.total_acc = 0
        self.count = 0
        self.threshold = threshold
    def forward(self, output, label):
        acc = (output > self.threshold).float() == label
        mean_acc = acc.float().mean()
        self.update(mean_acc)
        return mean_acc
    def update(self, metric):
        self.total_acc += metric
        self.count += 1
    def compute(self):
        return self.total_acc / self.count
    def reset(self):
        self.total_acc = 0
        self.count = 0


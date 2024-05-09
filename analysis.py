import torch
import numpy as np
from tqdm import tqdm
import sys

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


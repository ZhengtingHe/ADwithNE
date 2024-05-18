import sys
import datetime
import numpy as np
# from tqdm.rich import tqdm, trange
from tqdm import tqdm
import pandas as pd
import torch

sys.path.append("..")

# device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "mps" if sys.platform == "darwin" else "cpu"


def printlog(info):
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % now_time)
    print(str(info) + "\n")


class StepRunner:
    def __init__(self, net, loss_fn, dist_fn,
                 stage="train", metrics_dict=None,
                 optimizer=None
                 ):
        self.model, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer = optimizer
        self.dist_fn = dist_fn

    def step(self, source_event, target_event, emd):
        # forward
        source_event, target_event, emd = source_event.to(device), target_event.to(device), emd.to(device)
        source_embedding = self.model(source_event)
        target_embedding = self.model(target_event)
        emb_distance = self.dist_fn(source_embedding, target_embedding)
        # loss
        loss = self.loss_fn(emb_distance, emd)

        # backward
        if self.optimizer is not None and self.stage == "train":
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # metrics
        step_metrics = {self.stage + "_" + name: (metric_fn(emb_distance, emd)).item()
                        for name, metric_fn in self.metrics_dict.items()}
        return loss.item(), step_metrics

    def train_step(self, source_event, target_event, emd):
        self.model.train()
        return self.step(source_event, target_event, emd)

    @torch.no_grad()
    def eval_step(self, source_event, target_event, emd):
        self.model.eval()
        return self.step(source_event, target_event, emd)

    def __call__(self, source_event, target_event, emd):
        if self.stage == "train":
            return self.train_step(source_event, target_event, emd)
        else:
            return self.eval_step(source_event, target_event, emd)


class EpochRunner:
    def __init__(self, step_runner, verbose=True):
        self.step_runner = step_runner
        self.stage = step_runner.stage
        self.verbose = verbose

    def __call__(self, dataloader):
        total_loss, step = 0, 0
        if self.verbose:
            loop = tqdm(enumerate(dataloader), total=len(dataloader), file=sys.stdout)
        else:
            loop = enumerate(dataloader)
        epoch_log = {}
        for batch_idx, (source_event, target_event, emd) in loop:
            loss, step_metrics = self.step_runner(source_event, target_event, emd)
            step_log = dict({self.stage + "_loss": loss}, **step_metrics)
            total_loss += loss
            step += 1
            if batch_idx != len(dataloader) - 1:
                if self.verbose:
                    loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss / step
                epoch_metrics = {self.stage + "_" + name: metric_fn.compute().item()
                                 for name, metric_fn in self.step_runner.metrics_dict.items()}
                epoch_log = dict({self.stage + "_loss": epoch_loss}, **epoch_metrics)
                if self.verbose:
                    loop.set_postfix(**epoch_log)

                for name, metric_fn in self.step_runner.metrics_dict.items():
                    metric_fn.reset()
        return epoch_log


# @torch.compile()
def train_model(net, optimizer,
                loss_fn, dist_fn,
                metrics_dict,
                train_dataloader, val_dataloader=None,
                scheduler=None,
                epochs=10, ckpt_path='checkpoint.pt',
                patience=5, monitor="train_MAPE", mode="min", verbose=True):
    history = {}

    for epoch in range(1, epochs + 1):
        if verbose:
            printlog("Epoch {0} / {1}".format(epoch, epochs))

        # 1，train -------------------------------------------------  
        train_step_runner = StepRunner(net=net, stage="train", dist_fn=dist_fn,
                                       loss_fn=loss_fn, metrics_dict=metrics_dict,
                                       optimizer=optimizer)
        train_epoch_runner = EpochRunner(train_step_runner, verbose=verbose)
        train_metrics = train_epoch_runner(train_dataloader)

        for name, metric in train_metrics.items():
            history[name] = history.get(name, []) + [metric]

        # 2，validate -------------------------------------------------
        if val_dataloader:
            val_step_runner = StepRunner(net=net, stage="val",
                                         loss_fn=loss_fn, dist_fn=dist_fn, metrics_dict=metrics_dict)
            val_epoch_runner = EpochRunner(val_step_runner, verbose=verbose)
            with torch.no_grad():
                val_metrics = val_epoch_runner(val_dataloader)
            val_metrics["epoch"] = epoch
            for name, metric in val_metrics.items():
                history[name] = history.get(name, []) + [metric]

            # 3，early-stopping -------------------------------------------------
        arr_scores = history[monitor]
        best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
        if best_score_idx == len(arr_scores) - 1:
            torch.save(net.state_dict(), ckpt_path)
            print("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
                                                              arr_scores[best_score_idx]))
        if len(arr_scores) - best_score_idx > patience:
            print("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
                monitor, patience))
            break
        if scheduler:
            scheduler.step()
    net.load_state_dict(torch.load(ckpt_path))
    return pd.DataFrame(history)


@torch.no_grad()
def inference(model, dataloader, embed_dim=2):
    model.eval()
    embed = np.zeros((len(dataloader.dataset), embed_dim))
    for i, event in enumerate(tqdm(dataloader)):
        event = event.to(device)
        output = model(event)
        embed[i * len(event):(i + 1) * len(event)] = output.cpu().numpy()
    return embed

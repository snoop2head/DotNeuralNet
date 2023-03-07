from typing import Tuple, List, Dict, Any
import importlib

import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
from torch.functional import F
import torchvision.models as models
from torch.optim import AdamW, Optimizer
from transformers import get_scheduler
import pytorch_lightning as pl


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    """
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    """
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        # print('\nset_true: {0}'.format(set_true))
        # print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / float(
                len(set_true.union(set_pred))
            )
        # print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


class BrailleTagger(pl.LightningModule):
    def __init__(
        self,
        model_name="efficientnet_v2_s",
        num_category=6,  # one-hot encoded label [0, 0, 0, 0, 0, 0] ~ [1, 1, 1, 1, 1, 1]
        weights="EfficientNet_V2_S_Weights.IMAGENET1K_V1",
        n_training_steps=None,
        n_warmup_steps=None,
    ):
        super().__init__()

        self.efficientnet = importlib.import_module("torchvision.models.efficientnet")
        self.model = getattr(self.efficientnet, model_name)(weights=weights)
        del self.model.classifier

        self.classifier = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(1280, num_category),
        )

        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps

    def forward(self, x, labels=None):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        loss = F.multilabel_soft_margin_loss(x, labels)
        # multilabel classification accuracy
        accuracy = hamming_score(
            labels.cpu().detach().numpy(), x.cpu().detach().numpy() > 0.5
        )
        exact_match = metrics.accuracy_score(
            labels.cpu().detach().numpy(), x.cpu().detach().numpy() > 0.5
        )
        return {"loss": loss, "accuracy": accuracy, "exact_match": exact_match}

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], idx: int
    ) -> torch.Tensor:
        metrics = self(*batch)
        self.log_dict({f"train/{k}": v for k, v in metrics.items()})
        return metrics["loss"]

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], idx: int
    ) -> torch.Tensor:
        metrics = self(*batch)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()})
        return metrics["loss"]

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], idx: int
    ) -> torch.Tensor:
        metrics = self(*batch)
        self.log_dict({f"test/{k}": v for k, v in metrics.items()})
        return metrics["loss"]

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[dict[str, Any]]]:
        do_decay = [p for p in self.parameters() if p.requires_grad and p.ndim >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.ndim < 2]
        param_groups = [{"params": do_decay}, {"params": no_decay, "weight_decay": 0.0}]

        optimizer = AdamW(param_groups, lr=1e-4, weight_decay=0.01)
        scheduler = get_scheduler(
            optimizer=optimizer,
            name="linear",
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

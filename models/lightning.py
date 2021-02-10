#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Lightning wrapper for model, to facilitate easy training

-- Based on code by Pooya Ashtari
-- Adapted by Wouter Durnez
"""

import torch.nn.functional as F
from torch import nn
from torch import optim
from pytorch_lightning.core import LightningModule
from losses import dice_loss, dice_metric, dice_et, dice_tc, dice_wt
from models.unet import UNet

class Model(LightningModule):
    def __init__(
        self,
        network=UNet,
        network_params=None,
        loss=dice_loss,
        metrics=(dice_metric, dice_et, dice_tc, dice_wt,),
        optimizer=optim.AdamW, # TODO: Why AdamW?
        scheduler=optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params=None,
        scheduler_config=None,
        inference=nn.Identity,
        inference_params=None,
    ):
        super().__init__()

        # Create network
        self.network_params = {
            'in_channels': 4,
            'out_channels': 3

        } if network_params is None else network_params
        self.net = network(**self.network_params)

        # Loss function
        self.loss = loss

        # Performance metrics
        self.metrics = metrics

        # Optimizer
        self.optimizer = optimizer
        self.optimizer_params = {
            'lr': 1e-4,
            'weight_decay': 1e-2}

        # Learning rate scheduler   # TODO: Figure this out
        if scheduler is not None:
            self.scheduler = scheduler
            self.scheduler_params = (
                {} if scheduler_params is None else scheduler_params
            )
            self.scheduler_config = (
                {} if scheduler_config is None else scheduler_config
            )

        # inference
        self.inference = inference
        self.inference_params = (
            {} if inference_params is None else inference_params
        )

        # save hyperparameters
        self.save_hyperparameters()

    def get_n_parameters(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.net.parameters(), **self.optimizer_params
        )
        if hasattr(self, "scheduler"):
            scheduler = self.scheduler(optimizer, **self.scheduler_params)
            config = (
                [optimizer],
                [{"scheduler": scheduler, **self.scheduler_config}],
            )
        else:
            config = [optimizer]

        return config

    def training_step(self, batch, batch_idx):
        x, y = batch["input"], batch["target"]
        # forward
        y_hat = self(x)
        # calculate loss
        loss = self.loss(y_hat, y, **self.loss_params)
        # add logging and calculate metrics
        self.log(f"train_{self.loss.__name__}", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["input"], batch["target"]
        # inference
        y_hat = self.inference(x, self, **self.inference_params)
        # calculate metrics
        output = {}
        for m, pars in zip(self.metrics, self.metrics_params):
            output[f"val_{m.__name__}"] = m(y_hat, y, **pars)
        return output

    def validation_epoch_end(self, outputs):
        metric_names = dict.fromkeys(outputs[0])
        for metric_name in metric_names:
            metric_total = 0.0
            for output in outputs:
                metric_total += output[metric_name]
            metric_value = metric_total / len(outputs)
            if hasattr(metric_value, "item"):
                metric_value = metric_value.item()
            self.log(metric_name, metric_value, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch["input"], batch["target"]
        # inference
        y_hat = self.inference(x, self, **self.inference_params)
        # calculate metrics
        output = {}
        for m, pars in zip(self.metrics, self.metrics_params):
            output[f"test_{m.__name__}"] = m(y_hat, y, **pars)
        return output

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

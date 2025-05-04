#!/usr/bin/env python3
"""Recipe for training a Vision Transformer on Tiny ImageNet-200."""

import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.logger import get_logger
from torch.utils.data import Subset
import numpy as np


logger = get_logger(__name__)


# Brain Class for Tiny ImageNet
class Imagenet100Brain(sb.core.Brain):
    def compute_forward(self, batch, stage):
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        if stage == sb.Stage.TRAIN:
            inputs, targets = self.hparams.mixup_fn(inputs, targets)
        
        outputs = self.modules.model(inputs)
        return outputs, targets

    def compute_objectives(self, predictions, batch, stage):
        outputs, soft_targets = predictions
        _, hard_targets = batch

        log_probs = self.hparams.log_softmax(outputs)
        
        if stage == sb.Stage.TRAIN:
            loss = self.hparams.soft_loss(log_probs, soft_targets)  # soft targets from mixup
       
            with torch.no_grad():
                # Convert hard labels to long and use nll_loss
                hard_loss = self.hparams.nll_loss(log_probs, hard_targets)
                self._hard_losses.append(hard_loss.item())
        else: 
            loss = self.hparams.nll_loss(log_probs, hard_targets)

        if stage != sb.Stage.TRAIN:
            self.acc_metric.append(log_probs, hard_targets.unsqueeze(0))
        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage == sb.Stage.TRAIN:
            self._hard_losses = []
        else:
            self.acc_metric = self.hparams.acc_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss}
        if stage != sb.Stage.TRAIN:
            stage_stats["ACC"] = self.acc_metric.summarize()

        if stage == sb.Stage.TRAIN:
            avg_hard = sum(self._hard_losses) / len(self._hard_losses)
            stage_stats["hard_loss"] = avg_hard
            self.train_stats = stage_stats
            
        if stage == sb.Stage.VALID:
            lr = self.hparams.noam_annealing.current_lr
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=self.hparams.avg_checkpoints,
            )
            # print(
            #     f"Epoch: {epoch}, lr: {lr:.2e} - "
            #     f"train loss: {self.train_stats['loss']:.2e} - "
            #     f"valid loss: {stage_stats['loss']:.2e}, valid acc: {stage_stats['ACC'] * 100:.2f}%"
            # )

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            self.hparams.noam_annealing(self.optimizer)


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Tiny ImageNet-specific normalization
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.480, 0.448, 0.398], std=[0.277, 0.269, 0.282]
            ),
        ]
    )

    train_full = ImageFolder(
        root=os.path.join(hparams["data_folder"], "train"), transform=transform
    )

    total_samples = len(train_full)

    print(total_samples)
    subset_size = int(hparams["dataset_fraction"] * total_samples)
    print(subset_size)

    # Fix seed for reproducibility
    rng = np.random.default_rng(seed=42)
    subset_indices = rng.choice(total_samples, size=subset_size, replace=False)

    train_set = Subset(train_full, subset_indices)

    valid_set = ImageFolder(
        root=os.path.join(hparams["data_folder"], "val"), transform=transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=hparams["batch_size"],
        shuffle=True,
        num_workers=hparams["num_workers"],
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=hparams["batch_size"],
        shuffle=False,
        num_workers=hparams["num_workers"],
        drop_last=True
    )

    brain = Imagenet100Brain(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    brain.fit(
        brain.hparams.epoch_counter,
        train_set=train_loader,
        valid_set=valid_loader,
    )

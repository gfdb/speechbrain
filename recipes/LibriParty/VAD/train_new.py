#!/usr/bin/env python3
"""
Recipe for training a Voice Activity Detection (VAD) model on LibriParty.
No on-the-fly data augmentation is used in this variant.

To run an experiment:

python train_noaug.py hparams/train_noaug_adamw.yaml \
--data_folder=/path/to/LibriParty

Authors
 * Mohamed Kleit 2021
 * Arjun V 2021
 * Mirco Ravanelli 2021
"""

import sys

import numpy as np
import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class VADBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Given an input batch it computes the binary probability."""
        batch = batch.to(self.device)
        wavs, lens = batch.signal
        targets, lens_targ = batch.target
        self.targets = targets

        # From wav input to output binary prediction
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        feats = feats.detach()
        outputs = self.modules.cnn(feats)

        outputs = outputs.reshape(
            outputs.shape[0],
            outputs.shape[1],
            outputs.shape[2] * outputs.shape[3],
        )

        outputs, h = self.modules.rnn(outputs)
        outputs = self.modules.dnn(outputs)
        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the binary CE"
        predictions, lens = predictions
        targets = self.targets

        predictions = predictions[:, : targets.shape[-1], 0]

        loss = self.hparams.compute_BCE_cost(predictions, targets, lens)

        self.train_metrics.append(batch.id, torch.sigmoid(predictions), targets)
        if stage != sb.Stage.TRAIN:
            self.valid_metrics.append(
                batch.id, torch.sigmoid(predictions), targets
            )
        return loss

    def on_stage_start(self, stage, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        self.train_metrics = self.hparams.train_stats()
        if stage != sb.Stage.TRAIN:
            self.valid_metrics = self.hparams.test_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a stage."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            summary = self.valid_metrics.summarize(threshold=0.5)

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats={"loss": stage_loss, "summary": summary},
            )
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_loss, "summary": summary},
                num_to_keep=1,
                min_keys=["loss"],
                name=f"epoch_{epoch}",
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "summary": summary},
            )


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    # 1. Declarations:
    data_folder = hparams["data_folder"]
    train = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["annotation_train"],
        replacements={"data_root": data_folder},
    )
    validation = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["annotation_valid"],
        replacements={"data_root": data_folder},
    )
    test = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["annotation_test"],
        replacements={"data_root": data_folder},
    )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("signal")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("speech")
    @sb.utils.data_pipeline.provides("target")
    def vad_targets(speech, hparams=hparams):
        boundaries = (
            [
                (
                    int(interval[0] / hparams["time_resolution"]),
                    int(interval[1] / hparams["time_resolution"]),
                )
                for interval in speech
            ]
            if len(speech) > 0
            else []
        )
        gt = torch.zeros(
            int(
                np.ceil(
                    hparams["example_length"] * (1 / hparams["time_resolution"])
                )
            )
        )
        for indxs in boundaries:
            start, stop = indxs
            gt[start:stop] = 1
        return gt

    # Create dataset
    datasets = [train, validation, test]
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.add_dynamic_item(datasets, vad_targets)
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "signal", "target", "speech"]
    )

    # Split dataset
    train_data, valid_data, test_data = datasets
    return train_data, valid_data, test_data


# Begin Recipe!
if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from libriparty_prepare import prepare_libriparty

    # LibriParty preparation
    run_on_main(
        prepare_libriparty,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_json_folder": hparams["save_folder"],
            "sample_rate": hparams["sample_rate"],
            "window_size": hparams["example_length"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Dataset IO prep: creating Dataset objects
    train_data, valid_data, test_data = dataio_prep(hparams)

    # Trainer initialization
    vad_brain = VADBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training/validation loop
    vad_brain.fit(
        vad_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Test
    vad_brain.evaluate(
        test_data,
        min_key="loss",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

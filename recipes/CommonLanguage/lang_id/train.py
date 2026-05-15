#!/usr/bin/env python3
import os
import sys

from common_language_prepare import prepare_common_language
from hyperpyyaml import load_hyperpyyaml
import torch.nn.functional as F

import speechbrain as sb
from speechbrain.dataio import audio_io
from speechbrain.utils.logger import get_logger

"""Recipe for training a LID system with CommonLanguage.

To run this recipe, do the following:
> python train.py hparams/train_ecapa_tdnn.yaml

Author
------
 * Mirco Ravanelli 2021
 * Pavlo Ruban 2021
"""

logger = get_logger(__name__)


# Brain class for Language ID training
class LID(sb.Brain):
    def _kl_mode(self):
        """Return the configured consistency loss mode."""
        return str(
            getattr(self.hparams, "kl_mode", "clean_teacher")
        ).lower()

    def _kl_view_config(self):
        """Return multi-view layout metadata for KL consistency training."""
        candidates = []

        for augment_name in ("wav_augment", "fea_augment"):
            if not hasattr(self.hparams, augment_name):
                continue

            augment = getattr(self.hparams, augment_name)

            if hasattr(augment, "_views") or (
                augment_name == "wav_augment"
                and hasattr(self.hparams, "views")
            ):
                num_views = getattr(augment, "_views", None)
                if num_views is None:
                    num_views = getattr(self.hparams, "views")
                include_original = getattr(
                    augment, "_include_original", None
                )
                if include_original is None:
                    include_original = getattr(
                        self.hparams, "include_original", False
                    )
                candidates.append(
                    {
                        "name": augment_name,
                        "num_views": int(num_views),
                        "has_clean": bool(include_original),
                    }
                )
                continue

            if hasattr(augment, "batch_multiplier") or hasattr(
                augment, "concat_original"
            ):
                batch_multiplier = max(
                    1, int(getattr(augment, "batch_multiplier", 1))
                )
                concat_original = bool(
                    getattr(augment, "concat_original", False)
                )
                candidates.append(
                    {
                        "name": augment_name,
                        "num_views": batch_multiplier
                        + int(concat_original),
                        "has_clean": concat_original,
                    }
                )

        candidates = [
            candidate
            for candidate in candidates
            if candidate["num_views"] > 1
        ]
        if not candidates:
            raise ValueError(
                "KL loss requires a multi-view wav_augment or fea_augment."
            )
        if len(candidates) > 1:
            names = ", ".join(candidate["name"] for candidate in candidates)
            raise ValueError(
                "KL loss found multiple multi-view augmenters "
                f"({names}); please use only one KL view source."
            )

        config = candidates[0]
        if self._kl_mode() == "clean_teacher" and not config["has_clean"]:
            raise ValueError(
                "KL loss requires an included clean view. Set "
                "include_original=True for Wav2AugViews or "
                "concat_original=True for NewAugmenter."
            )
        if self._kl_mode() not in {"clean_teacher", "bidirectional"}:
            raise ValueError(
                "kl_mode must be 'clean_teacher' or 'bidirectional'."
            )

        return config

    def prepare_features(self, wavs, stage):
        """Prepare the features for computation, including augmentation.

        Arguments
        ---------
        wavs : tuple
            Input signals (tensor) and their relative lengths (tensor).
        stage : sb.Stage
            The current stage of training.

        Returns
        -------
        feats : torch.Tensor
            Computed features.
        lens : torch.Tensor
            The length of the corresponding features.
        """
        wavs, lens = wavs

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, lens = self.hparams.wav_augment(wavs, lens)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm_input(feats, lens)
        
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "fea_augment"):
            feats, lens = self.hparams.fea_augment(feats, lens)

        return feats, lens

    def compute_forward(self, batch, stage):
        """Runs all the computation of that transforms the input into the
        output probabilities over the N classes.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : torch.Tensor
            torch.Tensor that contains the posterior probabilities over the N classes.
        """

        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)

        # Compute features, embeddings and output
        feats, lens = self.prepare_features(batch.sig, stage)
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, inputs, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        inputs : tensors
            The output tensors from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        predictions, lens = inputs

        targets = batch.language_encoded.data

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "wav_augment"):
                targets = self.hparams.wav_augment.replicate_labels(targets)
                lens = self.hparams.wav_augment.replicate_labels(lens) 
            if hasattr(self.hparams, "fea_augment"):
                targets = self.hparams.fea_augment.replicate_labels(targets)
                lens = self.hparams.fea_augment.replicate_labels(lens)
            
            if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)

        loss = self.hparams.compute_cost(predictions, targets)

        kl_weight = float(getattr(self.hparams, "kl_weight", 0.0))
        if stage == sb.Stage.TRAIN and kl_weight != 0.0:
            view_config = self._kl_view_config()
            num_views = view_config["num_views"]
            total_batch = predictions.size(0)
            if total_batch % num_views != 0:
                raise ValueError(
                    f"KL loss expects a batch divisible by {num_views} views, "
                    f"but got {total_batch} prediction rows."
                )

            original_batch_size = total_batch // num_views
            view_logits = [
                predictions[
                    view_idx * original_batch_size : (view_idx + 1)
                    * original_batch_size
                ].squeeze(1)
                for view_idx in range(num_views)
            ]

            if self._kl_mode() == "clean_teacher":
                clean = view_logits[0]
                dirty_views = view_logits[1:]
                p_clean = F.softmax(clean, dim=-1).detach()
                kl_loss = sum(
                    F.kl_div(
                        F.log_softmax(dirty, dim=-1),
                        p_clean,
                        reduction="batchmean",
                    )
                    for dirty in dirty_views
                ) / len(dirty_views)
            else:
                pair_losses = []
                for left_idx in range(num_views):
                    for right_idx in range(left_idx + 1, num_views):
                        left = view_logits[left_idx]
                        right = view_logits[right_idx]
                        pair_losses.append(
                            0.5
                            * (
                                F.kl_div(
                                    F.log_softmax(left, dim=-1),
                                    F.softmax(right, dim=-1),
                                    reduction="batchmean",
                                )
                                + F.kl_div(
                                    F.log_softmax(right, dim=-1),
                                    F.softmax(left, dim=-1),
                                    reduction="batchmean",
                                )
                            )
                        )
                kl_loss = sum(pair_losses) / len(pair_losses)

            loss = loss + kl_weight * kl_loss

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, targets, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {
                    "epoch": epoch,
                    "lr": old_lr,
                    'steps': steps,
                    'optimizer': optimizer
                },
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `prepare_common_language` to have been called before this,
    so that the `train.csv`, `dev.csv`,  and `test.csv` manifest files
    are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "dev" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, lang01: 0, lang02: 1, ..)
    language_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig, _ = audio_io.load(wav)
        sig = sig.transpose(0, 1).squeeze(1)

        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("language")
    @sb.utils.data_pipeline.provides("language", "language_encoded")
    def label_pipeline(language):
        yield language
        language_encoded = language_encoder.encode_label_torch(language)
        yield language_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    for dataset in ["train", "dev", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"{dataset}_csv"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "language_encoded"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    language_encoder_file = os.path.join(
        hparams["save_folder"], "language_encoder.txt"
    )
    language_encoder.load_or_create(
        path=language_encoder_file,
        from_didatasets=[datasets["train"]],
        output_key="language",
    )

    return datasets, language_encoder


# Recipe begins!
if __name__ == "__main__":
    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.
    sb.utils.distributed.run_on_main(
        prepare_common_language,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )
    # Data preparation for augmentation
    if 'prepare_noise_data' in hparams:
        sb.utils.distributed.run_on_main(hparams["prepare_noise_data"])
    if 'prepare_rir_data' in hparams:
        sb.utils.distributed.run_on_main(hparams["prepare_rir_data"])

    # Create dataset objects "train", "dev", and "test" and language_encoder
    datasets, language_encoder = dataio_prep(hparams)

    # Fetch and load pretrained modules
    hparams["pretrainer"].collect_files()
    hparams["pretrainer"].load_collected()

    # Initialize the Brain object to prepare for mask training.
    lid_brain = LID(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    lid_brain.fit(
        epoch_counter=lid_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["dev"],
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = lid_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )

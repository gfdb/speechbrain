#!/usr/bin/env python3
"""Recipe for IEMOCAP emotion recognition with optional multi-view KL.

This is the consolidated entrypoint for wav2vec2 internal SpecAugment views,
Wav2Aug waveform views, and optional KL consistency.

Authors
 * Yingzhi WANG 2021
 * (spec-views adaptation)
"""

import os
import sys

import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb


class EmoIdBrain(sb.Brain):
    def _kl_mode(self):
        """Return the configured consistency loss mode."""
        return str(
            getattr(self.hparams, "kl_mode", "clean_teacher")
        ).lower()

    def _kl_view_config(self):
        """Return multi-view layout metadata for KL consistency training."""
        if hasattr(self.hparams, "batch_multiplier"):
            batch_multiplier = max(1, int(self.hparams.batch_multiplier))
            concat_original = bool(
                getattr(self.hparams, "concat_original", False)
            )
            if self._kl_mode() == "clean_teacher" and not concat_original:
                raise ValueError(
                    "KL loss requires an included clean view. Use "
                    "concat_original=True or include_original=True."
                )
            if self._kl_mode() not in {"clean_teacher", "bidirectional"}:
                raise ValueError(
                    "kl_mode must be 'clean_teacher' or 'bidirectional'."
                )
            return {
                "num_views": batch_multiplier + int(concat_original),
                "has_clean": concat_original,
            }

        candidates = []
        for augment_name in ("wav_augment",):
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
                "KL loss requires a multi-view wav2vec2 SpecAugment or "
                "wav_augment setup."
            )
        if len(candidates) > 1:
            raise ValueError(
                "KL loss found multiple multi-view augmenters; please use "
                "only one KL view source."
            )
        if self._kl_mode() == "clean_teacher" and not candidates[0]["has_clean"]:
            raise ValueError(
                "KL loss requires an included clean view. Use "
                "concat_original=True or include_original=True."
            )
        if self._kl_mode() not in {"clean_teacher", "bidirectional"}:
            raise ValueError(
                "kl_mode must be 'clean_teacher' or 'bidirectional'."
            )

        return candidates[0]

    def compute_forward(self, batch, stage):
        """Run wav2vec2, optionally creating multiple train-time views."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        if hasattr(self.hparams, "batch_multiplier"):
            if stage == sb.Stage.TRAIN:
                w2v_cfg = self.modules.wav2vec2.model.config
                view_outputs = []
                view_lens = []

                # Clean view first so KL can use the first batch block as teacher.
                if getattr(self.hparams, "concat_original", False):
                    w2v_cfg.apply_spec_augment = False
                    view_outputs.append(
                        self.modules.wav2vec2(wavs, lens)
                    )
                    view_lens.append(lens)

                # Augmented copies share one wav2vec2 call; each row samples its
                # own internal time/channel masks.
                batch_multiplier = max(1, int(self.hparams.batch_multiplier))
                if batch_multiplier > 0:
                    w2v_cfg.apply_spec_augment = True
                    aug_wavs = wavs.repeat(batch_multiplier, 1)
                    aug_lens = lens.repeat(batch_multiplier)
                    view_outputs.append(
                        self.modules.wav2vec2(aug_wavs, aug_lens)
                    )
                    view_lens.append(aug_lens)

                outputs = torch.cat(view_outputs, dim=0)
                lens = torch.cat(view_lens, dim=0)
            else:
                self.modules.wav2vec2.model.config.apply_spec_augment = False
                outputs = self.modules.wav2vec2(wavs, lens)
        else:
            if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
                wavs, lens = self.hparams.wav_augment(wavs, lens)

            outputs = self.modules.wav2vec2(wavs, lens)

        # Pooling  -> (B*V, D)  or  (B, D)
        outputs = self.hparams.avg_pool(outputs, lens)
        outputs = outputs.view(outputs.shape[0], -1)

        # Classifier head
        logits = self.modules.output_mlp(outputs)
        log_probs = self.hparams.log_softmax(logits)
        return log_probs, logits

    def compute_objectives(self, predictions, batch, stage):
        """Compute NLL loss, replicating labels for multi-view training."""
        emoid, _ = batch.emo_encoded
        emoid = emoid.squeeze(1)

        log_probs, logits = predictions

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "batch_multiplier"):
            batch_multiplier = max(1, int(self.hparams.batch_multiplier))
            total_views = batch_multiplier + int(
                getattr(self.hparams, "concat_original", False)
            )
            emoid = emoid.repeat(total_views)
        elif stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "wav_augment"):
                emoid = self.hparams.wav_augment.replicate_labels(emoid)

        loss = self.hparams.compute_cost(log_probs, emoid)

        kl_weight = float(getattr(self.hparams, "kl_weight", 0.0))
        if stage == sb.Stage.TRAIN and kl_weight != 0.0:
            view_config = self._kl_view_config()
            total_views = view_config["num_views"]
            total_batch = logits.size(0)
            if total_batch % total_views != 0:
                raise ValueError(
                    f"KL loss expects a batch divisible by {total_views} "
                    f"views, but got {total_batch} logits rows."
                )

            original_batch_size = total_batch // total_views
            view_logits = [
                logits[
                    view_idx * original_batch_size : (view_idx + 1)
                    * original_batch_size
                ]
                for view_idx in range(total_views)
            ]

            if self._kl_mode() == "clean_teacher":
                clean = view_logits[0]
                dirty_views = view_logits[1:]
                if not dirty_views:
                    raise ValueError(
                        "KL loss requires at least one augmented/dirty view."
                    )

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
                for left_idx in range(total_views):
                    for right_idx in range(left_idx + 1, total_views):
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

            loss = loss + (kl_weight * kl_loss)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, log_probs, emoid)

        return loss

    # ------------------------------------------------------------------
    # Stage bookkeeping (identical to the base recipe)
    # ------------------------------------------------------------------
    def on_stage_start(self, stage, epoch=None):
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        else:
            stats = {
                "loss": stage_loss,
                "error_rate": self.error_metrics.summarize("average"),
            }

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stats["error_rate"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            old_lr_w2v, new_lr_w2v = self.hparams.lr_annealing_wav2vec2(
                stats["error_rate"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec2_optimizer, new_lr_w2v
            )

            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr, "wav2vec2_lr": old_lr_w2v},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stats, min_keys=["error_rate"]
            )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def init_optimizers(self):
        """Separate optimizers for wav2vec2 and classifier head."""
        self.wav2vec2_optimizer = self.hparams.wav2vec2_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec2_opt", self.wav2vec2_optimizer
            )
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

        self.optimizers_dict = {
            "model_optimizer": self.optimizer,
            "wav2vec2_optimizer": self.wav2vec2_optimizer,
        }


def dataio_prep(hparams):
    """Prepare IEMOCAP datasets (identical to the base recipe)."""

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    @sb.utils.data_pipeline.takes("emo")
    @sb.utils.data_pipeline.provides("emo", "emo_encoded")
    def label_pipeline(emo):
        yield emo
        emo_encoded = label_encoder.encode_label_torch(emo)
        yield emo_encoded

    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "emo_encoded"],
        )

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="emo",
    )
    return datasets


def main():
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from iemocap_prepare import prepare_data  # noqa E402

    if not hparams["skip_prep"]:
        sb.utils.distributed.run_on_main(
            prepare_data,
            kwargs={
                "data_original": hparams["data_folder"],
                "save_json_train": hparams["train_annotation"],
                "save_json_valid": hparams["valid_annotation"],
                "save_json_test": hparams["test_annotation"],
                "split_ratio": hparams["split_ratio"],
                "different_speakers": hparams["different_speakers"],
                "test_spk_id": hparams["test_spk_id"],
                "seed": hparams["seed"],
            },
        )

    if "prepare_noise_data" in hparams:
        sb.utils.distributed.run_on_main(hparams["prepare_noise_data"])

    datasets = dataio_prep(hparams)

    hparams["wav2vec2"] = hparams["wav2vec2"].to(device=run_opts["device"])
    if not hparams["freeze_wav2vec2"] and hparams["freeze_wav2vec2_conv"]:
        hparams["wav2vec2"].model.feature_extractor._freeze_parameters()

    emo_id_brain = EmoIdBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    emo_id_brain.fit(
        epoch_counter=emo_id_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    test_stats = emo_id_brain.evaluate(
        test_set=datasets["test"],
        min_key="error_rate",
        test_loader_kwargs=hparams["dataloader_options"],
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Recipe for training emotion recognition with wav2vec2 SpecAugment views.

Creates multiple views of each batch by toggling the internal SpecAugment
inside HuggingFace wav2vec2.  For example, with include_clean=True and
spec_aug_views=4 you get:

    view 0  ->  wav2vec2 forward with SpecAugment OFF   (clean)
    view 1  ->  wav2vec2 forward with SpecAugment ON    (random mask 1)
    view 2  ->  wav2vec2 forward with SpecAugment ON    (random mask 2)
    view 3  ->  wav2vec2 forward with SpecAugment ON    (random mask 3)

All four representations are pooled, projected, and their losses averaged.

Key CLI overrides
-----------------
    --spec_aug_views 4       total number of views (clean + augmented)
    --num_clean_views 1      how many of those views skip SpecAugment

Examples
--------
    # 1 clean + 3 augmented
    python train_spec_views.py hparams/spec_aug_views.yaml \\
        --data_folder /path/to/IEMOCAP --spec_aug_views 4 --num_clean_views 1

    # 2 clean, no augmented (vanilla concat)
    python train_spec_views.py hparams/spec_aug_views.yaml \\
        --data_folder /path/to/IEMOCAP --spec_aug_views 2 --num_clean_views 2

    # 2 augmented, no clean
    python train_spec_views.py hparams/spec_aug_views.yaml \\
        --data_folder /path/to/IEMOCAP --spec_aug_views 2 --num_clean_views 0

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
    def compute_forward(self, batch, stage):
        """Run wav2vec2 multiple times with / without SpecAugment."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # ---- During training, build multi-view representations ----
        if stage == sb.Stage.TRAIN:
            total_views = self.hparams.spec_aug_views
            n_clean = self.hparams.num_clean_views
            n_aug = total_views - n_clean

            w2v_cfg = self.modules.wav2vec2.model.config
            view_outputs = []

            # Clean views — batched forward pass, SpecAugment OFF
            if n_clean > 0:
                w2v_cfg.apply_spec_augment = False
                clean_wavs = wavs.repeat(n_clean, 1)         # (B*n_clean, T_wav)
                clean_lens = lens.repeat(n_clean)             # (B*n_clean,)
                view_outputs.append(
                    self.modules.wav2vec2(clean_wavs, clean_lens)
                )

            # Augmented views — batched forward pass, SpecAugment ON;
            # _compute_mask_indices draws independent masks per sample,
            # so repeat(n, 1) in a single pass == n separate passes.
            if n_aug > 0:
                w2v_cfg.apply_spec_augment = True
                aug_wavs = wavs.repeat(n_aug, 1)             # (B*n_aug, T_wav)
                aug_lens = lens.repeat(n_aug)                 # (B*n_aug,)
                view_outputs.append(
                    self.modules.wav2vec2(aug_wavs, aug_lens)
                )

            # (B*n_clean, T, D) + (B*n_aug, T, D) -> (B*V, T, D)
            outputs = torch.cat(view_outputs, dim=0)         # (B*V, T, D)
            lens = lens.repeat(total_views)                   # (B*V,)
        else:
            # Validation / test: single clean forward pass
            self.modules.wav2vec2.model.config.apply_spec_augment = False
            outputs = self.modules.wav2vec2(wavs, lens)

        # Pooling  -> (B*V, D)  or  (B, D)
        outputs = self.hparams.avg_pool(outputs, lens)
        outputs = outputs.view(outputs.shape[0], -1)

        # Classifier head
        outputs = self.modules.output_mlp(outputs)
        log_probs = self.hparams.log_softmax(outputs)
        return log_probs, outputs 

    def compute_objectives(self, predictions, batch, stage):
        """Compute NLL loss, replicating labels for multi-view training."""
        emoid, _ = batch.emo_encoded
        emoid = emoid.squeeze(1)

        log_probs, logits = predictions

        if stage == sb.Stage.TRAIN:
            total_views = self.hparams.spec_aug_views
            emoid = emoid.repeat(total_views)

        loss = self.hparams.compute_cost(log_probs, emoid)
        
        # assumes final bs will be 4
        clean = logits[0]
        dirty1 = logits[1]
        dirty2 = logits[2]
        dirty3 = logits[3]

        p_clean = F.softmax(clean, dim=-1).detach()
        
        kl_loss = F.kl_div(F.log_softmax(dirty1, dim=-1), p_clean, reduction="batchmean")
        kl_loss += F.kl_div(F.log_softmax(dirty2, dim=-1), p_clean, reduction="batchmean")
        kl_loss += F.kl_div(F.log_softmax(dirty3, dim=-1), p_clean, reduction="batchmean")

        kl_loss = kl_loss / 3

        loss = loss + (self.hparams.kl_weight * kl_loss)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, emoid)

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


# RECIPE BEGINS!
if __name__ == "__main__":
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

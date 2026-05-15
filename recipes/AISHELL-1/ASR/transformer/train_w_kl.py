#!/usr/bin/env/python3
"""

AISHELL-1 transformer model recipe. (Adapted from the LibriSpeech recipe.)

Authors
    * Jianyuan Zhong 2021
    * Titouan Parcollet 2021
"""

import sys

import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


# Define training procedure
class ASR(sb.core.Brain):
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

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
            tokens_bos = self.hparams.wav_augment.replicate_labels(tokens_bos)

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav2aug_gpu"):
            wavs, wav_lens = self.hparams.wav2aug_gpu(wavs, wav_lens)
            
        # compute features
        feats = self.hparams.compute_features(wavs)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "fea_augment"):
            feats, wav_lens = self.hparams.fea_augment(feats, wav_lens)
            tokens_bos = self.hparams.fea_augment.replicate_labels(tokens_bos)

        current_epoch = self.hparams.epoch_counter.current
        feats = self.hparams.normalize(feats, wav_lens, epoch=current_epoch)

        # forward modules
        src = self.modules.CNN(feats)
        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        )

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        hyps = None
        current_epoch = self.hparams.epoch_counter.current
        is_valid_search = (
            stage == sb.Stage.VALID
            and current_epoch % self.hparams.valid_search_interval == 0
        )
        is_test_search = stage == sb.Stage.TEST

        if is_valid_search:
            hyps, _, _, _ = self.hparams.valid_search(
                enc_out.detach(), wav_lens
            )
        elif is_test_search:
            hyps, _, _, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        return p_ctc, p_seq, wav_lens, hyps, pred

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (p_ctc, p_seq, wav_lens, hyps, seq_logits) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if stage == sb.Stage.TRAIN:
            # Labels must be extended if parallel augmentation or concatenated
            # augmentation was performed on the input (increasing the time dimension)
            if hasattr(self.hparams, "wav_augment"):
                tokens = self.hparams.wav_augment.replicate_labels(tokens)
                tokens_lens = self.hparams.wav_augment.replicate_labels(tokens_lens)
                tokens_eos = self.hparams.wav_augment.replicate_labels(tokens_eos)
                tokens_eos_lens = self.hparams.wav_augment.replicate_labels(tokens_eos_lens)

            if hasattr(self.hparams, "fea_augment"):
                tokens = self.hparams.fea_augment.replicate_labels(tokens)
                tokens_lens = self.hparams.fea_augment.replicate_labels(tokens_lens)
                tokens_eos = self.hparams.fea_augment.replicate_labels(tokens_eos)
                tokens_eos_lens = self.hparams.fea_augment.replicate_labels(tokens_eos_lens)


        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )
        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )

        if stage == sb.Stage.TRAIN and float(self.hparams.kl_weight) != 0.0:
            view_config = self._kl_view_config()
            num_views = view_config["num_views"]
            total_batch = seq_logits.size(0)
            if total_batch % num_views != 0:
                raise ValueError(
                    f"KL loss expects a batch divisible by {num_views} views, "
                    f"but got {total_batch} seq-logit rows."
                )

            original_batch_size = total_batch // num_views
            view_logits = [
                seq_logits[
                    view_idx * original_batch_size : (view_idx + 1)
                    * original_batch_size
                ]
                for view_idx in range(num_views)
            ]

            U = view_logits[0].size(1)

            lens_steps_by_view = []
            for view_idx in range(num_views):
                start = view_idx * original_batch_size
                end = (view_idx + 1) * original_batch_size
                lens_rel = tokens_eos_lens[start:end]
                lens_steps_by_view.append(
                    (lens_rel * U).round().long().clamp(min=1, max=U)
                )

            step_ids = torch.arange(U, device=view_logits[0].device)[None, :]

            def masked_kl(
                student_logits: torch.Tensor,
                target_logits: torch.Tensor,
                lens_steps: torch.Tensor,
                detach_target: bool,
            ) -> torch.Tensor:
                target_probs = F.softmax(target_logits, dim=-1)
                if detach_target:
                    target_probs = target_probs.detach()
                log_p_student = F.log_softmax(student_logits, dim=-1)
                kl_per_step = F.kl_div(
                    log_p_student, target_probs, reduction="none"
                ).sum(dim=-1)
                mask = (step_ids < lens_steps[:, None]).float()
                return (kl_per_step * mask).sum() / mask.sum().clamp_min(1.0)

            if self._kl_mode() == "clean_teacher":
                clean = view_logits[0]
                dirty_views = view_logits[1:]
                kl_loss = sum(
                    masked_kl(
                        dirty,
                        clean,
                        lens_steps_by_view[0],
                        detach_target=True,
                    )
                    for dirty in dirty_views
                ) / len(dirty_views)
            else:
                pair_losses = []
                for left_idx in range(num_views):
                    for right_idx in range(left_idx + 1, num_views):
                        left = view_logits[left_idx]
                        right = view_logits[right_idx]
                        lens_steps = torch.minimum(
                            lens_steps_by_view[left_idx],
                            lens_steps_by_view[right_idx],
                        )
                        pair_losses.append(
                            0.5
                            * (
                                masked_kl(
                                    left,
                                    right,
                                    lens_steps,
                                    detach_target=False,
                                )
                                + masked_kl(
                                    right,
                                    left,
                                    lens_steps,
                                    detach_target=False,
                                )
                            )
                        )
                kl_loss = sum(pair_losses) / len(pair_losses)

            loss = loss + float(self.hparams.kl_weight) * kl_loss


        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval

            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                # Decode token terms to words
                predicted_words = [
                    tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
                ]
                target_words = [wrd.split(" ") for wrd in batch.wrd]
                if self.hparams.remove_spaces:
                    predicted_words = ["".join(p) for p in predicted_words]
                    target_words = ["".join(t) for t in target_words]
                    self.cer_metric.append(ids, predicted_words, target_words)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        return loss

    def on_fit_batch_start(self, batch, should_step):
        """Gets called at the beginning of each fit_batch."""
        # check if we need to switch optimizer
        # if so change the optimizer from Adam to SGD
        self.check_and_reset_optimizer()

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            self.hparams.noam_annealing(self.optimizer)

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.cer_metric = self.hparams.cer_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID:
            # report different epoch stages according current stage
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.stage_one_epochs:
                lr = self.hparams.noam_annealing.current_lr
                steps = self.hparams.noam_annealing.n_steps
                optimizer = self.optimizer.__class__.__name__
            else:
                lr = self.hparams.lr_sgd
                steps = -1
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

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.cer_file, "w", encoding="utf-8") as w:
                self.cer_metric.write_stats(w)

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )

    def check_and_reset_optimizer(self):
        """reset the optimizer if training enters stage 2"""
        current_epoch = self.hparams.epoch_counter.current
        if not hasattr(self, "switched"):
            self.switched = False
            if isinstance(self.optimizer, torch.optim.SGD):
                self.switched = True

        if self.switched is True:
            return

        if current_epoch > self.hparams.stage_one_epochs:
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)

            self.switched = True

    def on_fit_start(self):
        """Initialize the right optimizer on the training start"""
        super().on_fit_start()

        # if the model is resumed from stage two, reinitialize the optimizer
        current_epoch = self.hparams.epoch_counter.current
        current_optimizer = self.optimizer
        if current_epoch > self.hparams.stage_one_epochs:
            del self.optimizer
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            # Load latest checkpoint to resume training if interrupted
            if self.checkpointer is not None:
                # do not reload the weights if training is interrupted right before stage 2
                group = current_optimizer.param_groups[0]
                if "momentum" not in group:
                    return

                self.checkpointer.recover_if_possible()

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint average if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts,
            recoverable_name="model",
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration", reverse=True)

    datasets = [train_data, valid_data, test_data]

    # Defining tokenizer and loading it
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]

        train_batch_sampler = DynamicBatchSampler(
            train_data, **dynamic_hparams, length_func=lambda x: x["duration"]
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data, **dynamic_hparams, length_func=lambda x: x["duration"]
        )

    return (
        train_data,
        valid_data,
        test_data,
        tokenizer,
        train_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # 1.  # Dataset prep (parsing Librispeech)
    from aishell_prepare import prepare_aishell  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_aishell,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "skip_prep": hparams["skip_prep"],
            "remove_compressed_wavs": hparams["remove_compressed_wavs"],
        },
    )
    if "prepare_noise_data" in hparams:
        run_on_main(hparams["prepare_noise_data"])

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_data,
        tokenizer,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams)

    # We download and pretrain the tokenizer
    hparams["pretrainer"].collect_files()
    hparams["pretrainer"].load_collected()

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    asr_brain.tokenizer = tokenizer

    # Changing the samplers if dynamic batching is activated
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }
    if valid_bsampler is not None:
        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    # Testing
    asr_brain.evaluate(
        test_data, test_loader_kwargs=hparams["test_dataloader_opts"]
    )

import torch
import torch.nn as nn
import inspect
from speechbrain.utils.callchains import lengths_arg_exists
import torch.nn.functional as F


class NewAugmenter(nn.Module):
    """A memory-efficient faster implementation for batch augmentation.

    Args:
        augmentations (list): a list of augmentation functions that operate on a batch
        min_num_aug (int, optional): The lower bound for sampling the number of augmentations.
        num_aug (int, optional): The upper bound of number of augmentations to be applied to the batch and copies of it (if any). Defaults to 1.
        batch_multiplier (int, optional): The number of times to multiply the batch. Defaults to 0.
        concat_original (bool, optional): Concatenates the original batch after augmentation is done. Increases `batch_size` by `batch_size`. Defaults to False.
        aug_strategy (str, optional):
            - If aug_strategy is "random", each batch copy gets a random number (up to num_aug)
                of randomly sampled augmentations.
            - If aug_strategy is "all", each batch copy gets all the augmentations applied sequentially, num_aug times,
                per batch copy. 
            Defaults to "random".
    """
    def __init__(
        self,
        augmentations: list,
        min_num_aug: int = 1,
        num_aug: int = 1,
        batch_multiplier: int = 0,
        concat_original: bool = False,
        aug_strategy: str = "random"  # "random" or "all"
    ):
        super().__init__()
        self.min_num_aug = min_num_aug
        self.num_aug = num_aug
        self.batch_multiplier = batch_multiplier if batch_multiplier > 1 else 1
        self.augmentations = augmentations
        self.concat_original = concat_original
        self.aug_strategy = aug_strategy.lower()
            

        # check which augmentation functions require lengths argument
        self.require_lengths = {}
        for i, aug in enumerate(self.augmentations):
            if hasattr(aug, 'forward'):
                self.require_lengths[i] = lengths_arg_exists(aug.forward)
            else:
                # for function-based augmentations
                self.require_lengths[i] = lengths_arg_exists(aug)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None):
        """
        Args:
            x (torch.Tensor): Shape [batch_size, ...].
            lengths (torch.Tensor, optional): Shape [batch_size].
        Returns:
            - If lengths is None:
                A tensor of shape [ (batch_multiplier + concat_original) * batch_size, ... ]
            - If lengths is not None:
                A tuple (augmented_tensor, augmented_lengths)
        """
        device = x.device
        original_bs = x.shape[0]

        # create list of batch copies
        batch_copies = []
        length_copies = []
        for _ in range(self.batch_multiplier):
            batch_copies.append(x.clone())
            if lengths is not None:
                length_copies.append(lengths.clone())

        if self.aug_strategy == "random":
            # random number of augmentation steps for each batch copy
            sampled_num_aug = torch.randint(
                low=self.min_num_aug,
                high=self.num_aug + 1,
                size=(self.batch_multiplier,),
                device=device
            )  # shape: [batch_multiplier]

            # Precompute random augmentation indices for every augmentation step and batch copy.
            # This creates a matrix of shape [num_aug, batch_multiplier] where each entry
            # is a random augmentation index.
            precomputed_rand_indices = torch.randint(
                low=0,
                high=len(self.augmentations),
                size=(self.num_aug, self.batch_multiplier),
                device=device
            )

            # Sequentially apply augmentation steps.
            for aug_step in range(self.num_aug):
                # Determine which batch copies still require augmentation at this step.
                apply_mask = aug_step < sampled_num_aug
                if not apply_mask.any():
                    break

                # Get indices of batch copies that need augmentation.
                indices_to_augment = torch.where(apply_mask)[0]
                # Use precomputed random indices for the current augmentation step.
                random_aug_indices = precomputed_rand_indices[aug_step, indices_to_augment]

                # Apply the sampled augmentations.
                for idx_tensor, aug_idx_tensor in zip(indices_to_augment, random_aug_indices):
                    idx = idx_tensor.item()
                    aug_idx = aug_idx_tensor.item()
                    augmentation_fn = self.augmentations[aug_idx]
                    if self.require_lengths[aug_idx] and lengths is not None:
                        result = augmentation_fn(batch_copies[idx], lengths=length_copies[idx])
                        if isinstance(result, tuple):
                            batch_copies[idx], length_copies[idx] = result
                        else:
                            batch_copies[idx] = result
                    else:
                        batch_copies[idx] = augmentation_fn(batch_copies[idx])

        elif self.aug_strategy == "all":
            # apply every augmentation sequentially for each batch copy copy
            for idx, augmentation_fn in enumerate(self.augmentations):
                for i in range(self.batch_multiplier):
                    if self.require_lengths[idx] and lengths is not None:
                        result = augmentation_fn(batch_copies[i], lengths=length_copies[i])
                        if isinstance(result, tuple):
                            batch_copies[i], length_copies[i] = result
                        else:
                            batch_copies[i] = result
                    else:
                        batch_copies[i] = augmentation_fn(batch_copies[i])
        else:
            raise ValueError(f"Unsupported aug_strategy: {self.aug_strategy}")

        # optionally concatenate the original batch
        if self.concat_original:
            batch_copies.append(x)
            if lengths is not None:
                length_copies.append(lengths)
        
        # if there is a single batch copy
        # just return it as all the lengths will be the same
        if len(batch_copies) == 1:
            return batch_copies[0], length_copies[0]
        
        return self.concatenate_outputs(batch_copies, length_copies)

    def replicate_labels(self, labels, deep_copy=False):
        """Replicate labels based on how many copies of the
        batch were made.

        Args:
            labels (torch.Tensor): The labels for your data.
            deep_copy (bool, optional):
                - If False, copy given labels without allocating new memory. 
                - If True, make deep copies.
            Defaults to False.

        Returns:
            torch.Tensor: The labels for the augmented data.
        """
        batch_size = labels.shape[0]
        non_batch_dims = labels.shape[1:]
        num_repl = self.batch_multiplier + int(self.concat_original)
        
        if num_repl == 1:
            return labels

        if deep_copy:
            out = torch.cat([labels.clone() for _ in range(num_repl)], dim=0)
        else:
            expanded = labels.unsqueeze(1).expand(batch_size, num_repl, *non_batch_dims)

            # reshape but in non-contiguous blocks
            out = expanded.reshape(batch_size * num_repl, *non_batch_dims)

            # expanding copied the labels, however it did it sequentially
            # i.e. element of batch 1 is copied num_repl times at pos 0,1,2,...
            # we need to reorder to line up the labels correctly
            indices = torch.arange(batch_size * num_repl).reshape(batch_size, num_repl).transpose(0, 1).reshape(-1)
            
            # make contiguous to optimize training speed
            out = out[indices].contiguous()
        return out


    def concatenate_outputs(self, augment_lst, augment_len_lst):
        """
        Concatenate a list of augmented signals, accounting for varying temporal lengths.
        Padding is applied to ensure all signals can be concatenated.

        Arguments
        ---------
        augment_lst : List of torch.Tensor
            List of augmented signals to be concatenated.
        augment_len_lst : List of torch.Tensor
            List of lengths corresponding to the augmented signals.

        Returns
        -------
        concatenated_signals : torch.Tensor
            A tensor containing the concatenated signals.
        concatenated_lengths : torch.Tensor
            A tensor containing the concatenated signal lengths.

        Notes
        -----
        This function takes a list of augmented signals, which may have different temporal
        lengths due to variations such as speed changes. It pads the signals to match the
        maximum temporal dimension found among the input signals and rescales the lengths
        accordingly before concatenating them.
        """
        # Find the maximum temporal dimension (batch length) among the sequences
        max_len = max(augment.shape[1] for augment in augment_lst)

        # if all augmentations already have the maximum length, concatenate directly
        # without adding padding
        if all(augment.shape[1] == max_len for augment in augment_lst):
            output = torch.cat(augment_lst, dim=0)
            lens = torch.cat(augment_len_lst, dim=0)
            return output, lens

        # Rescale the sequence lengths to adjust for augmented batches with different temporal dimensions.
        augment_len_lst = [
            length * (output.shape[1] / max_len)
            for length, output in zip(augment_len_lst, augment_lst)
        ]

        # Pad sequences to match the maximum temporal dimension.
        # Note that some augmented batches, like those with speed changes, may have different temporal dimensions.
        augment_lst = [
            F.pad(output, (0, max_len - output.shape[1]))
            for output in augment_lst
        ]

        # Concatenate the padded sequences and rescaled lengths
        output = torch.cat(augment_lst, dim=0)
        output_lengths = torch.cat(augment_len_lst, dim=0)

        return output, output_lengths


"""Classes for implementing data augmentation pipelines.

Authors
 * Mirco Ravanelli 2022
"""

import logging
import random
from typing import Literal

import torch
import torch.nn.functional as F

from speechbrain.utils.callchains import lengths_arg_exists
from .augmenter import Augmenter
from typing import List, Union, Tuple

logger = logging.getLogger(__name__)


class AugmentBlock(torch.nn.Module):
    def __init__(
        self,
        augment_type: Literal["parallel", "sequential"],
        concat_original: bool = False,
        min_augmentations: int = 1,
        max_augmentations: int = None,
        shuffle_augmentations: bool = False,
        repeat_block: int = 1,
        augment_start_index: int = 0,
        augment_end_index: int = None,
        concat_start_index: int = 0,
        concat_end_index: int = None,
        block_prob: float = 1.0,
        augmentations: list = [],
        enable_augmentations: list = None,
    ):
        self.augment_type = augment_type
        self.concat_original = concat_original
        self.min_augmentations = min_augmentations
        self.max_augmentations = max_augmentations
        self.shuffle_augmentations = shuffle_augmentations
        self.augment_start_index = augment_start_index
        self.augment_end_index = augment_end_index
        self.concat_start_index = concat_start_index
        self.concat_end_index = concat_end_index
        self.repeat_block = repeat_block
        self.block_prob = block_prob
        self.augmentations = augmentations
        self.enable_augmentations = enable_augmentations
        # Check min and max augmentations
        Augmenter.check_min_max_augmentations(self)

        # This variable represents the total number of augmentations to perform for each signal,
        # including the original signal in the count.
        self.num_augmentations = None
        self.do_augment = True


        # Check repeat augment arguments
        if not isinstance(self.repeat_augment, int):
            raise ValueError("repeat_augment must be an integer.")

        if self.repeat_augment < 0:
            raise ValueError("repeat_augment must be greater than 0.")

        if enable_augmentations is None:
            enable_augmentations = [True] * len(augmentations)
        elif not isinstance(enable_augmentations, list):
            raise ValueError("enable_augmentations must be a list.")
        elif len(enable_augmentations) != len(augmentations):
            raise ValueError(
                "enable_augmentations must have the same length as augmentations."
            )

        augmentations = [
            aug
            for aug, enabled in zip(augmentations, enable_augmentations)
            if enabled
        ]

        # TODO: other validation and logger warnings


        # Check if augmentation modules need the length argument
        self.require_lengths = {}
        for aug_key, aug_fun in self.augmentations.items():
            self.require_lengths[aug_key] = lengths_arg_exists(aug_fun.forward)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        self.do_augment = True
        if random.random() > self.block_prob:
            self.do_augment = False
            return x, lengths

        x_original = x
        len_original = lengths

        # Determine the ending index for augmentation, considering user-specified or default values.
        self.augment_end_index_batch = (
            min(self.augment_end_index, x.shape[0])
            if self.augment_end_index is not None
            else x.shape[0]
        )

        # If the augmentation starting index is beyond the size of the data, return the original data.
        if self.augment_start_index >= x.shape[0]:
            self.do_augment = False
            logger.warning(
                "No augmentation is applied because the augmentation start index is greater than or equal to the number of examples in the input batch."
            )
            return x, lengths
    
         # Select the portion of the input to augment and update lengths accordingly.
        x = x[self.augment_start_index : self.augment_end_index_batch]

        lengths = lengths[
            self.augment_start_index : self.augment_end_index_batch
        ]
        
        # Lists to collect the outputs
        output_lst = []
        output_len_lst = []

        # Concatenate the original signal if required
        self._concatenate_original(
            x_original,
            len_original,
            output_lst,
            output_len_lst
        )

        # Perform augmentations
        for i in range(self.repeat_block):
            output, output_lengths = self.augment(x, lengths)
            output_lst.append(output)
            output_len_lst.append(output_lengths)

        # Concatenate the final outputs while handling scenarios where
        # different temporal dimensions may arise due to augmentations
        # like speed change.
        output, output_lengths = self.concatenate_outputs(
            output_lst, output_len_lst
        )

        return output, output_lengths




    def augment(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        lengths: Union[torch.Tensor, List[torch.Tensor]]
    ):
        outputs, output_lens = [], []
        
        if isinstance(x, list):
            multi_out, multi_lens = self._augment_multi(
                x, lengths
            )
        else:
            out, lens = self._augment_single(
                x, lengths
            )
        
        for batch, lens in zip(x, lengths):
            num_augmentations = torch.randint(
                low=self.min_augmentations,
                high=self.max_augmentations + 1,
                size=(1,),
                device=batch.device,
            )

            # Get augmentations list
            augmentations_lst = list(self.augmentations.keys())

            # No augmentation
            if (
                self.repeat_augment == 0
                or self.N_augment == 0
                or len(augmentations_lst) == 0
            ):
                self.do_augment = False
                return batch, lengths

            # Shuffle augmentation
            if self.shuffle_augmentations:
                random.shuffle(augmentations_lst)
            
            selected_augmentations = augmentations_lst[0 : num_augmentations]
            
            for aug_obj, aug_name in selected_augmentations:
                if self.augment_type == "parallel":
                    aug_outs, aug_lens = self._apply_parallel_augmentations(
                        batch, lengths, aug_obj
                    )
                elif self.augment_type == "sequential":
                    aug_outs, aug_lens = self._apply_sequential_augmentations(
                        batch, lengths, aug_obj
                    )
                outputs.append(aug_outs)
                output_lens.append(aug_lens)

    def _augment_multi(
        self,
        x: List[torch.Tensor],
        lengths: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor]]:
        # apply augmentation to each batch given
        # this happens when a parallel block
        # is followed by a sequential block
        out_lst, out_lens_lst = [], []
        for batch, batch_lens in zip(x, lengths):
            aug_out, aug_out_lens = self._augment_single(
                batch, batch_lens
            )


    def _augment_single(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        
        num_augmentations = torch.randint(
            low=self.min_augmentations,
            high=self.max_augmentations + 1,
            size=(1,),
            device=x.device,
        )

        # Get augmentations list
        augmentations_lst = list(self.augmentations.keys())

        # No augmentation
        if (
            self.repeat_augment == 0
            or self.N_augment == 0
            or len(augmentations_lst) == 0
        ):
            self.do_augment = False
            return x, lengths

        # Shuffle augmentation
        if self.shuffle_augmentations:
            random.shuffle(augmentations_lst)
        
        selected_augmentations = augmentations_lst[0 : num_augmentations]
        
        outputs, output_lens = [], []
        for aug_obj, aug_name in selected_augmentations:
            if self.augment_type == "parallel":
                # this can return a list of tensors
                aug_outs, aug_lens = self._apply_parallel_augmentations(
                    x, lengths, aug_obj, aug_name
                )
            elif self.augment_type == "sequential":
                aug_outs, aug_lens = self._apply_sequential_augmentations(
                    x, lengths, aug_obj, aug_name
                )
            outputs.append(aug_outs)
            output_lens.append(aug_lens)
        
        if self.augment_type == "parallel" and len(selected_augmentations) > 1:
            # TODO: handle flattenning for parallel output
            flat_outs, flat_lens = [], []
            for output in outputs:
                if isinstance(output, list):
                    

    
    def _concatenate_original(
            self,
            x: torch.Tensor,
            lengths: torch.Tensor,
            output_lst: list,
            output_len_lst: list
        ):
        # Concatenate the original signal if required
        self.skip_concat = not (self.concat_original)

        if self.concat_original:
            # Check start index
            if self.concat_start_index >= x.shape[0]:
                self.skip_concat = True
                pass
            else:
                self.skip_concat = False
                # Determine the ending index for concatenation, considering user-specified or default values.
                self.concat_end_index_batch = (
                    min(self.concat_end_index, x.shape[0])
                    if self.concat_end_index is not None
                    else x.shape[0]
                )

                output_lst.append(
                    x[
                        self.concat_start_index : self.concat_end_index_batch
                    ]
                )
                output_len_lst.append(
                    lengths[
                        self.concat_start_index : self.concat_end_index_batch
                    ]
                )
    def _apply_parallel_augmentations(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        lengths: Union[torch.Tensor, List[torch.Tensor]],
        aug_obj: Union[torch.nn.Module, 'AugmentBlock'],
        aug_name: str
    ) -> Tuple[List[torch.Tensor]]:
        output_list = []
        output_len_list = []

        # if `x` is a list of tensors, apply augmentations
        # apply augmentation logic to each tensor in the list
        if isinstance(x, list):
            pass
        else:
            pass

    def _apply_sequential_augmentations(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        aug_obj: Union[torch.nn.Module, 'AugmentBlock'],
        aug_name: str
    ) -> Tuple[torch.Tensor]:

        # TODO: add slicing logic for aug_start/end
        kwargs = {}

        # add 'lengths' keyword argument if required
        # for this augmentation
        if self.require_lengths[aug_name]:
            kwargs['lengths'] = lengths #[idx]
        
        out, lens = aug_obj(x, **kwargs)
        return out, lens
        
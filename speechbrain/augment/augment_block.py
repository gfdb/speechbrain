
"""Classes for implementing data augmentation pipelines.

Authors
 * Mirco Ravanelli 2022
 * Gianfranco Dumoulin Bertucci 2024
"""

import logging
import random
from typing import Literal

import torch
import torch.nn.functional as F

from speechbrain.utils.callchains import lengths_arg_exists
from typing import List, Union, Tuple
from .augment_helpers import concatenate_outputs, check_min_max_augmentations


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
        augmentations: List[torch.nn.Module] = [],
        enable_augmentations: List[bool] = None,
        flatten_output: bool = True
    ):
        super().__init__()
        self.augment_type = augment_type
        self.concat_original = concat_original
        self.augmentations = augmentations
        self.min_augmentations = min_augmentations
        self.max_augmentations = max_augmentations
        self.shuffle_augmentations = shuffle_augmentations
        self.augment_start_index = augment_start_index
        self.augment_end_index = augment_end_index
        self.concat_start_index = concat_start_index
        self.concat_end_index = concat_end_index
        self.repeat_block = repeat_block
        self.block_prob = block_prob
        self.flatten_output = flatten_output
        check_min_max_augmentations(self)

        # This variable represents the total number of augmentations to perform for each signal,
        # including the original signal in the count.
        self.num_augmentations = None
        self.do_augment = True


        # Check repeat augment arguments
        if not isinstance(self.repeat_block, int):
            raise ValueError("repeat_block must be an integer.")

        if self.repeat_block < 0:
            raise ValueError("repeat_block must be greater than 0.")

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

        # Turn augmentations into a dictionary
        self.augmentations = {
            augmentation.__class__.__name__ + str(i): augmentation
            for i, augmentation in enumerate(augmentations)
        }

        # TODO: other validation and logger warnings


        # Check if augmentation modules need the length argument
        self.require_lengths = {}
        for aug_key, aug_fun in self.augmentations.items():
            self.require_lengths[aug_key] = lengths_arg_exists(aug_fun.forward)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        print('forward:')
        self.do_augment = True
        if random.random() > self.block_prob:
            self.do_augment = False
            return x, lengths
        print('\t self.flatten_output:', self.flatten_output)

        x_original = x
        len_original = lengths
        if isinstance(x, list):
            print('\t x is a list')
        else:
            print('\t x is NOT a list')

        # Lists to collect the outputs
        output_lst = []
        output_len_lst = []

        # this will concatenate the original signal if required
        self._concatenate_original(
            x_original,
            len_original,
            output_lst,
            output_len_lst
        )

        # Perform augmentations
        aug_out, aug_out_lens = x, lengths
        for _ in range(self.repeat_block):
            print('repeating...')
            aug_out, aug_out_lens = self.augment(aug_out, aug_out_lens)
            # print('\t aug_out.shape:', aug_out.shape)
            # print('\t aug_out_lens.shape:', aug_out_lens.shape)

        print('\t type(aug_out):', type(aug_out))
        if isinstance(aug_out, list):
            output_lst.extend(aug_out)
            output_len_lst.extend(aug_out_lens)
        else:
            output_lst.append(aug_out)
            output_len_lst.append(aug_out_lens)

        if self.flatten_output:
            # Concatenate the final outputs while handling scenarios where
            # different temporal dimensions may arise due to augmentations
            # like speed change.
            if any(isinstance(i, list) for i in output_lst):
                output_lst = self.flatten_nested_list(output_lst)
                output_len_lst = self.flatten_nested_list(output_len_lst)
            # print('\t type(output_lst[0]):', type(output_lst[0]))
            # print('\t type(output_len_lst[0]):', type(output_len_lst[0]))
            # print('\t len(output_lst):', len(output_lst))
            # print('\t output_len_lst.shape:', output_len_lst.shape)

            output_lst, output_len_lst = concatenate_outputs(
                output_lst, output_len_lst
            )
        print('\t type(output_lst):', type(output_lst))
        print('\t len(output_lst):', len(output_lst))
        print('\t type(output_lst[0]):', type(output_lst[0]))
        print('\t output_lst[0].shape:', output_lst[0].shape)
        print("==============")
        print('forward complete')
        print("==============")
        return output_lst, output_len_lst


    def augment(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        lengths: Union[torch.Tensor, List[torch.Tensor]]
    ):
        print('augment:')
        if isinstance(x, list):
            print('\t x is a list')
            return self._augment_multi(x, lengths)
        print('\t x is NOT a list')
        return self._augment_single_batch(x, lengths)

    

    def _augment_multi(
        self,
        x: List[torch.Tensor],
        lengths: List[torch.Tensor]
    ) -> Union[Tuple[torch.Tensor], Tuple[List[torch.Tensor]]]:
        print('_augment_multi')
        print('\t type(x):', type(x))
        print('\t len(x):', len(x))
        # Apply augmentation to each batch given.
        # This will happen when a parallel block
        # is followed by a sequential block.
        out_lst, out_lens_lst = [], []
        for batch, batch_lens in zip(x, lengths):
            # print('batch_lens', batch_lens)
            if isinstance(batch, list):
                raise ValueError('batch should not be a list')
                print('\t batch is a list')
                print(f'\t len: {len(batch)}')
            aug_out, aug_out_lens = self._augment_single_batch(
                batch, batch_lens
            )
            # if a list is returned (parallel and n_aug > 1)
            if isinstance(aug_out, list):
                print('\t aug_out is a list')
                out_lst.extend(aug_out)
                out_lens_lst.extend(aug_out_lens)
            else:
                print('\t aug_out is NOT a list')
                out_lst.append(aug_out)
                out_lens_lst.append(aug_out_lens)

        print('+++++++++++++++++++++++++')
        print('_augment_multi complete')        
        print('+++++++++++++++++++++++++')
        return out_lst, out_lens_lst


    def _augment_single_batch(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
    ) -> Union[Tuple[torch.Tensor], Tuple[List[torch.Tensor]]]:
        """Augments a single batch of data in parallel or in
        sequence.

        Args:
            x (torch.Tensor): _description_
            lengths (torch.Tensor): _description_

        Returns:
            Union[Tuple[torch.Tensor], Tuple[List[torch.Tensor]]]: If parallel and num_aug > 1, a tuple of list of
        tensors will be returned, else if parallel and num_aug = 1 or
        sequential, a tuple of tensors will be returned.
        """
        print('_augment_single_batch')
        print('\t x shape:', x.shape)
        # print('lengths:', lengths)
        
        num_augmentations = torch.randint(
            low=self.min_augmentations,
            high=self.max_augmentations + 1,
            size=(1,),
            device=x.device,
        ).item()

        print('\t num_augmentations:', num_augmentations)

        # Get augmentations list
        augmentations_lst = list(self.augmentations.keys())

        # No augmentation
        if (
            self.repeat_block == 0
            or num_augmentations == 0
            or len(augmentations_lst) == 0
        ):
            self.do_augment = False
            return x, lengths

        # Shuffle augmentation
        if self.shuffle_augmentations:
            random.shuffle(augmentations_lst)
        
        selected_augmentations = augmentations_lst[0 : num_augmentations]
        print('\t selected_augmentations:', selected_augmentations)
        
        outputs, output_lens = [], [lengths]
        
        # if the augment type is sequential 
        # we will augment the 0th index
        # over and over again
        if self.augment_type == "sequential":
            outputs.append(x)
            

        for aug_key in selected_augmentations:
            print('\t', aug_key)
            # this can return a list (of tuples) of tensors
            if self.augment_type == "parallel":
                aug_outs, aug_lens = self._apply_augmentation(
                    x, lengths, self.augmentations[aug_key], aug_key
                )
                # print('aug_lens:', aug_lens)
                outputs.append(aug_outs)
                if aug_lens:
                    output_lens.append(aug_lens)
                else:
                    output_lens.append(lengths)
                # print('output_lens:', output_lens)

            elif self.augment_type == "sequential":
                # this will return a tuple of tensors
                # print('output_lens[0]:', output_lens[0])
                aug_outs, aug_lens = self._apply_augmentation(
                    outputs[0], output_lens[0], self.augmentations[aug_key], aug_key
                )
                # print('aug_outs:', aug_outs)
                # print('aug_lens:', aug_lens)
                
                outputs[0] = aug_outs
                if aug_lens:
                    output_lens[0] = aug_lens

                if aug_key == selected_augmentations[-1]:
                    # flatten if this is the last augmentation in the loop
                    outputs, output_lens = outputs[0], output_lens[0]
        
        print('\t len(outputs):', len(outputs))
        return outputs, output_lens
            
    
    
    def _concatenate_original(
            self,
            x: torch.Tensor,
            lengths: torch.Tensor,
            output_lst: list,
            output_len_lst: list
        ):
        # TODO: handle case where original is a list
        # consider 

        # Concatenate the original signal if required
        self.skip_concat = not (self.concat_original)

        if self.skip_concat:
            return
        
        # Check start index
        if self.concat_start_index >= x.shape[0]:
            self.skip_concat = True
            return
        
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

    def _apply_augmentation(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        aug_obj: torch.nn.Module,
        aug_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # print('_apply_augmentation')
        if isinstance(x, list):
            # print(x)
            raise ValueError('x is a list')
        
        # if isinstance(lengths, list):
            # print(lengths)
        #     raise ValueError('lengths is a list')
        
        out, out_lens = [], []
        # TODO: add slicing logic for aug_start/end
        kwargs = {}
        if self.require_lengths[aug_name]:
            # print(f"{aug_name} requires lengths")
            kwargs['lengths'] = lengths
        
        out = aug_obj(x, **kwargs)
        
        if isinstance(out, tuple):
            out, out_lens = out
        
        return out, out_lens
    
    def flatten_nested_list(self, nested_list):
        flat_list = []
        for item in nested_list:
            if isinstance(item, list):
                flat_list.extend(self.flatten_nested_list(item))
            elif isinstance(item, torch.Tensor):
                flat_list.append(item)
            else:
                raise TypeError(f"Unexpected type {type(item)} encountered. Expected list or torch.Tensor.")
        return flat_list
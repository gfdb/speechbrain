from typing import Union, List, TYPE_CHECKING
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .augment_block import AugmentBlock
    from .augmenter import Augmenter

def check_min_max_augmentations(augment_obj: Union['Augmenter', 'AugmentBlock']):
    """Checks the min_augmentations and max_augmentations arguments."""
    if augment_obj.min_augmentations is None:
        augment_obj.min_augmentations = 1
    if augment_obj.max_augmentations is None:
        augment_obj.max_augmentations = len(augment_obj.augmentations)
    if augment_obj.max_augmentations > len(augment_obj.augmentations):
        augment_obj.max_augmentations = len(augment_obj.augmentations)
    if augment_obj.min_augmentations > len(augment_obj.augmentations):
        augment_obj.min_augmentations = len(augment_obj.augmentations)


def concatenate_outputs(
    augment_lst: List[torch.Tensor], augment_len_lst: List[torch.Tensor]
):
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

    if augment_len_lst:
        # Rescale the sequence lengths to adjust for augmented batches with different temporal dimensions.
        augment_len_lst = [
            length * (output.shape[1] / max_len)
            for length, output in zip(augment_len_lst, augment_lst)
        ]
        augment_len_lst = torch.cat(augment_len_lst, dim=0)

    # Pad sequences to match the maximum temporal dimension.
    # Note that some augmented batches, like those with speed changes, may have different temporal dimensions.
    augment_lst = [
        F.pad(output, (0, max_len - output.shape[1])) for output in augment_lst
    ]

    # Concatenate the padded sequences and rescaled lengths
    augment_lst = torch.cat(augment_lst, dim=0)

    return augment_lst, augment_len_lst

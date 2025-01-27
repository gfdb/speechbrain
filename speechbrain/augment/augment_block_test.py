from time_domain import AddNoise, DropChunk, SpeedPerturb
from augmenter import Augmenter
from augment_block import AugmentBlock
import torch

add_noise = AddNoise()
drop_chunk = DropChunk()

# para_augment_block = AugmentBlock(
#     augment_type = "parallel",
#     concat_original=False,
#     min_augmentations=2,
#     max_augmentations=2,
#     shuffle_augmentations=True,
#     repeat_block=1,
#     augmentations=[add_noise, drop_chunk],
#     flatten_output=False
# )

# seq_augment_block = AugmentBlock(
#     augment_type = "sequential",
#     concat_original=False,
#     min_augmentations=2,
#     max_augmentations=2,
#     shuffle_augmentations=True,
#     repeat_block=1,
#     augmentations=[add_noise, drop_chunk],
#     flatten_output=True
# )

augmenter = Augmenter(
    parallel_augment=False,
    concat_original=True,
    min_augmentations=2,
    max_augmentations=2,
    shuffle_augmentations=False,
    augmentations=[add_noise, drop_chunk]
)

# Generate test input
x = torch.randn(1, 16000)  # Single sample with 16000 length
lengths = torch.tensor([16000])

augmented_out, augmented_lens = augmenter(x, lengths)
print('augmented_out.size()', augmented_out.size())
print('augmented_lens.size()', augmented_lens.size())

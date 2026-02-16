"""Frequency-Domain Sequential Data Augmentation Classes

This module comprises classes tailored for augmenting sequential data in the
frequency domain, such as spectrograms and mel spectrograms.
Its primary purpose is to enhance the resilience of neural models during the training process.

Authors:
- Peter Plantinga (2020)
- Mirco Ravanelli (2023)
"""

import random

import torch
import torch.nn as nn


class SpectrogramDrop(nn.Module):
    def __init__(
        self,
        drop_length_low=5,
        drop_length_high=15,   # this corresponds to SpecAugment "T" (time) or "F" (freq)
        drop_count_low=1,
        drop_count_high=3,
        replace="zeros",
        dim=1,
        max_ratio=None,        # <-- this is SpecAugment "p" for time masking (e.g., 1.0 or 0.2)
    ):
        super().__init__()
        self.drop_length_low = drop_length_low
        self.drop_length_high = drop_length_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.replace = replace
        self.dim = dim
        self.max_ratio = max_ratio

        if drop_length_low > drop_length_high:
            raise ValueError("Low limit must not be more than high limit")
        if drop_count_low > drop_count_high:
            raise ValueError("Low limit must not be more than high limit")

        self.replace_opts = ["zeros", "mean", "rand", "cutcat", "swap", "random_selection"]
        if self.replace not in self.replace_opts:
            raise ValueError(f"Invalid 'replace' option. Select one of {', '.join(self.replace_opts)}")

        if self.max_ratio is not None:
            if not (0.0 < float(self.max_ratio) <= 1.0):
                raise ValueError("max_ratio (p) must be in (0, 1].")

    def forward(self, spectrogram: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # Manage 4D tensors
        if spectrogram.dim() == 4:
            spectrogram = spectrogram.view(-1, spectrogram.shape[2], spectrogram.shape[3])

        batch_size, time_duration, fea_size = spectrogram.shape
        D = time_duration if self.dim == 1 else fea_size

        # n_masks: same for all samples (matches common SpecAugment impls)
        n_masks = int(torch.randint(
            low=self.drop_count_low,
            high=self.drop_count_high + 1,
            size=(1,),
            device=spectrogram.device,
        ).item())
        if n_masks == 0:
            return spectrogram

        # Determine per-sample effective length tau_i (only relevant for time masking)
        if self.dim == 1 and lengths is not None:
            # lengths can be either relative (0..1) or absolute frames; handle both safely:
            if lengths.dtype.is_floating_point:
                tau = torch.clamp((lengths * time_duration).floor().long(), min=1, max=time_duration)
            else:
                tau = torch.clamp(lengths.long(), min=1, max=time_duration)
        else:
            tau = torch.full((batch_size,), D, device=spectrogram.device, dtype=torch.long)

        # Compute per-sample max allowed mask length: min(T, floor(p * tau_i))
        # Only apply the p cap for time masking. For freq masking, SpecAugment doesn't use p.
        if self.dim == 1 and self.max_ratio is not None:
            max_len = torch.minimum(
                torch.full((batch_size,), self.drop_length_high, device=spectrogram.device, dtype=torch.long),
                torch.clamp((tau.float() * float(self.max_ratio)).floor().long(), min=1),
            )
        else:
            max_len = torch.full((batch_size,), self.drop_length_high, device=spectrogram.device, dtype=torch.long)

        # If max_len < low for some samples, clamp so sampling remains valid
        low = int(self.drop_length_low)
        max_len = torch.clamp(max_len, min=low)

        # Sample mask lengths per (sample, mask)
        # randint high is exclusive, so use max_len+1
        u = torch.rand((batch_size, n_masks), device=spectrogram.device)
        mask_len = (low + torch.floor(u * (max_len.unsqueeze(1) - low + 1).float())).long()  # [B, M]

        # Sample start positions per (sample, mask): pos in [0, tau_i - mask_len]
        # Ensure non-negative range:
        max_pos = torch.clamp(tau.unsqueeze(1) - mask_len, min=0)  # [B, M]
        u2 = torch.rand((batch_size, n_masks), device=spectrogram.device)
        mask_pos = torch.floor(u2 * (max_pos + 1).float()).long()  # [B, M]

        # Build boolean mask on dimension D (padded width), but only mask within tau_i
        arange = torch.arange(D, device=spectrogram.device).view(1, 1, -1)  # [1,1,D]
        pos = mask_pos.unsqueeze(-1)  # [B,M,1]
        leng = mask_len.unsqueeze(-1)  # [B,M,1]
        m = (pos <= arange) & (arange < (pos + leng))  # [B,M,D]
        m = m.any(dim=1)  # [B,D]

        # Also ensure we don't mask beyond true length for each sample (time masking case)
        if self.dim == 1:
            valid = arange.squeeze(0).squeeze(0).unsqueeze(0) < tau.unsqueeze(1)  # [B,D]
            m = m & valid

        # Expand to spectrogram shape
        mask = m.unsqueeze(2) if self.dim == 1 else m.unsqueeze(1)  # [B,T,1] or [B,1,F]

        # Replacement (unchanged from your logic, but avoid mutating self.replace permanently)
        replace_mode = self.replace
        if replace_mode == "random_selection":
            replace_mode = random.choice(self.replace_opts[:-1])

        if replace_mode == "zeros":
            spectrogram = spectrogram.masked_fill(mask, 0.0)
        elif replace_mode == "mean":
            mean = spectrogram.mean().detach()
            spectrogram = spectrogram.masked_fill(mask, mean)
        elif replace_mode == "rand":
            mx = spectrogram.max().detach()
            mn = spectrogram.min().detach()
            r = torch.rand_like(spectrogram) * (mx - mn) + mn
            spectrogram = torch.where(mask, r, spectrogram)
        elif replace_mode == "cutcat":
            rolled = torch.roll(spectrogram, shifts=1, dims=0)
            spectrogram = torch.where(mask, rolled, spectrogram)
        elif replace_mode == "swap":
            shift = int(torch.randint(low=1, high=spectrogram.shape[1], size=(1,), device=spectrogram.device).item())
            rolled = torch.roll(spectrogram, shifts=shift, dims=1)
            spectrogram = torch.where(mask, rolled, spectrogram)

        return spectrogram


class Warping(torch.nn.Module):
    """
    Apply time or frequency warping to a spectrogram.

    If `dim=1`, time warping is applied; if `dim=2`, frequency warping is applied.
    This implementation selects a center and a window length to perform warping.
    It ensures that the temporal dimension remains unchanged by upsampling or
    downsampling the affected regions accordingly.

    Reference:
        https://arxiv.org/abs/1904.08779

    Arguments
    ---------
    warp_window : int, optional
        The width of the warping window. Default is 5.
    warp_mode : str, optional
        The interpolation mode for time warping. Default is "bicubic."
    dim : int, optional
        Dimension along which to apply warping (1 for time, 2 for frequency).
        Default is 1.

    Example
    -------
    >>> # Time-warping
    >>> warp = Warping()
    >>> spectrogram = torch.rand(4, 150, 40)
    >>> print(spectrogram.shape)
    torch.Size([4, 150, 40])
    >>> out = warp(spectrogram)
    >>> print(out.shape)
    torch.Size([4, 150, 40])
    >>> # Frequency-warping
    >>> warp = Warping(dim=2)
    >>> spectrogram = torch.rand(4, 150, 40)
    >>> print(spectrogram.shape)
    torch.Size([4, 150, 40])
    >>> out = warp(spectrogram)
    >>> print(out.shape)
    torch.Size([4, 150, 40])
    """

    def __init__(self, warp_window=5, warp_mode="bicubic", dim=1):
        super().__init__()
        self.warp_window = warp_window
        self.warp_mode = warp_mode
        self.dim = dim

    def forward(self, spectrogram):
        """
        Apply warping to the input spectrogram.

        Arguments
        ---------
        spectrogram : torch.Tensor
            Input spectrogram with shape `[batch, time, fea]`.

        Returns
        -------
        torch.Tensor
            Augmented spectrogram with shape `[batch, time, fea]`.
        """

        # Set warping dimension
        if self.dim == 2:
            spectrogram = spectrogram.transpose(1, 2)

        original_size = spectrogram.shape
        window = self.warp_window

        # 2d interpolation requires 4D or higher dimension tensors
        # x: (Batch, Time, Freq) -> (Batch, 1, Time, Freq)
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(1)

        len_original = spectrogram.shape[2]
        if len_original - window <= window:
            return spectrogram.view(*original_size)

        # Compute center and corresponding window
        c = torch.randint(window, len_original - window, (1,))[0]
        w = torch.randint(c - window, c + window, (1,))[0] + 1

        # Update the left part of the spectrogram
        left = torch.nn.functional.interpolate(
            spectrogram[:, :, :c],
            (w, spectrogram.shape[3]),
            mode=self.warp_mode,
            align_corners=True,
        )

        # Update the right part of the spectrogram.
        # When the left part is expanded, the right part is compressed by the
        # same factor, and vice versa.
        right = torch.nn.functional.interpolate(
            spectrogram[:, :, c:],
            (len_original - w, spectrogram.shape[3]),
            mode=self.warp_mode,
            align_corners=True,
        )

        # Injecting the warped left and right parts.
        spectrogram[:, :, :w] = left
        spectrogram[:, :, w:] = right
        spectrogram = spectrogram.view(*original_size)

        # Transpose if freq warping is applied.
        if self.dim == 2:
            spectrogram = spectrogram.transpose(1, 2)

        return spectrogram


class RandomShift(torch.nn.Module):
    """Shifts the input tensor by a random amount, allowing for either a time
    or frequency (or channel) shift depending on the specified axis.
    It is crucial to calibrate the minimum and maximum shifts according to the
    requirements of your specific task.
    We recommend using small shifts to preserve information integrity.
    Using large shifts may result in the loss of significant data and could
    potentially lead to misalignments with corresponding labels.

    Arguments
    ---------
    min_shift : int
        The minimum channel shift.
    max_shift : int
        The maximum channel shift.
    dim: int
        The dimension to shift.

    Example
    -------
    >>> # time shift
    >>> signal = torch.zeros(4, 100, 80)
    >>> signal[0, 50, :] = 1
    >>> rand_shift = RandomShift(dim=1, min_shift=-10, max_shift=10)
    >>> lengths = torch.tensor([0.2, 0.8, 0.9, 1.0])
    >>> output_signal, lengths = rand_shift(signal, lengths)

    >>> # frequency shift
    >>> signal = torch.zeros(4, 100, 80)
    >>> signal[0, :, 40] = 1
    >>> rand_shift = RandomShift(dim=2, min_shift=-10, max_shift=10)
    >>> lengths = torch.tensor([0.2, 0.8, 0.9, 1.0])
    >>> output_signal, lengths = rand_shift(signal, lengths)
    """

    def __init__(self, min_shift=0, max_shift=0, dim=1):
        super().__init__()
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.dim = dim

        # Check arguments
        if self.max_shift < self.min_shift:
            raise ValueError("max_shift must be  >= min_shift")

    def forward(self, waveforms, lengths):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`
        """
        # Pick a frequency to drop
        N_shifts = torch.randint(
            low=self.min_shift,
            high=self.max_shift + 1,
            size=(1,),
            device=waveforms.device,
        )
        waveforms = torch.roll(waveforms, shifts=N_shifts.item(), dims=self.dim)

        # Update lengths in the case of temporal shift.
        if self.dim == 1:
            lengths = lengths + N_shifts / waveforms.shape[self.dim]
            lengths = torch.clamp(lengths, min=0.0, max=1.0)

        return waveforms, lengths

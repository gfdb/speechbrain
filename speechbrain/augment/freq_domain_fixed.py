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
        mask_len: int,        # fixed T (time) or F (freq)
        num_masks: int,       # fixed mT or mF
        dim: int = 1,         # 1=time, 2=freq
        max_ratio: float | None = None,  # p (time masking only)
    ):
        super().__init__()

        if mask_len < 0:
            raise ValueError("mask_len must be >= 0")
        if num_masks < 0:
            raise ValueError("num_masks must be >= 0")
        if dim not in (1, 2):
            raise ValueError("dim must be 1 (time) or 2 (freq)")
        if max_ratio is not None and not (0.0 < float(max_ratio) <= 1.0):
            raise ValueError("max_ratio (p) must be in (0, 1]")

        self.mask_len = int(mask_len)
        self.num_masks = int(num_masks)
        self.dim = dim
        self.max_ratio = max_ratio

    def forward(
        self,
        spectrogram: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:

        orig_shape = spectrogram.shape

        # Accept [B,T,F] or [B,C,T,F]
        folded = False
        if spectrogram.dim() == 4:
            b, c, t, f = spectrogram.shape
            spectrogram = spectrogram.reshape(b * c, t, f)
            folded = True
        elif spectrogram.dim() != 3:
            raise ValueError("Input must be [B,T,F] or [B,C,T,F]")

        B, T, F = spectrogram.shape
        D = T if self.dim == 1 else F

        if self.num_masks == 0 or self.mask_len == 0:
            return spectrogram.reshape(orig_shape) if folded else spectrogram

        # Effective length (for time masking only)
        if self.dim == 1 and lengths is not None:
            if lengths.dtype.is_floating_point:
                tau = torch.clamp((lengths * T).floor().long(), min=1, max=T)
            else:
                tau = torch.clamp(lengths.long(), min=1, max=T)
        else:
            tau = torch.full((B,), D, device=spectrogram.device, dtype=torch.long)

        # Apply p cap if provided (time masking only)
        if self.dim == 1 and self.max_ratio is not None:
            max_len = torch.clamp(
                (tau.float() * float(self.max_ratio)).floor().long(),
                min=0,
            )
            mask_len = torch.minimum(
                torch.full_like(max_len, self.mask_len),
                max_len,
            )
        else:
            mask_len = torch.full((B,), self.mask_len,
                                  device=spectrogram.device,
                                  dtype=torch.long)

        arange = torch.arange(D, device=spectrogram.device).view(1, 1, -1)

        total_mask = torch.zeros((B, D),
                                 device=spectrogram.device,
                                 dtype=torch.bool)

        for _ in range(self.num_masks):

            # sample start uniformly in [0, tau - mask_len]
            max_pos = torch.clamp(tau - mask_len, min=0)
            start = torch.randint(
                low=0,
                high=(max_pos + 1),
                size=(B,),
                device=spectrogram.device,
            )

            pos = start.view(B, 1, 1)
            leng = mask_len.view(B, 1, 1)

            m = (pos <= arange) & (arange < (pos + leng))
            m = m.squeeze(1)

            # don't mask beyond true length (time case)
            if self.dim == 1:
                valid = arange.squeeze(0).squeeze(0).unsqueeze(0) < tau.unsqueeze(1)
                m = m & valid

            total_mask |= m

        if self.dim == 1:
            mask = total_mask.unsqueeze(2)  # [B,T,1]
        else:
            mask = total_mask.unsqueeze(1)  # [B,1,F]

        out = spectrogram.masked_fill(mask, 0.0)

        return out.reshape(orig_shape) if folded else out


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

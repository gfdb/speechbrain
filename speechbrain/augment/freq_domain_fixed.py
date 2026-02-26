"""Frequency-Domain Sequential Data Augmentation Classes

This module comprises classes tailored for augmenting sequential data in the
frequency domain, such as spectrograms and mel spectrograms.
Its primary purpose is to enhance the resilience of neural models during the training process.

Authors:
- Peter Plantinga (2020)
- Mirco Ravanelli (2023)
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn


def _to_frame_lengths(
    lengths: torch.Tensor | None,
    max_frames: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert relative or absolute lengths to absolute frame lengths.

    Args:
        lengths (torch.Tensor | None): If float and <= 1, treated as relative in [0, 1].
            Otherwise treated as absolute frame counts. Shape [B].
        max_frames (int): Total frames T in the input tensor.
        batch_size (int): Batch size B.
        device (torch.device): Target device.

    Returns:
        torch.Tensor: Absolute frame lengths [B] (long), clamped to [1, T].
    """
    if lengths is None:
        return torch.full((batch_size,), max_frames, dtype=torch.long, device=device)

    lengths = lengths.to(device)
    if lengths.numel() != batch_size:
        raise ValueError(f"Expected lengths with {batch_size} elements, got {lengths.numel()}.")

    if torch.is_floating_point(lengths) and torch.max(lengths) <= 1.0 + 1e-6:
        frame_lengths = torch.round(lengths * max_frames).long()
    else:
        frame_lengths = lengths.long()

    return torch.clamp(frame_lengths, min=1, max=max_frames)


class SpectrogramDrop(nn.Module):
    """SpecAugment-style masking (time or frequency), zero-fill only.

    Args:
        mask_len (int, optional): Maximum mask length (T for time, F for freq). Defaults to 15.
        num_masks (int, optional): Number of masks to apply. Defaults to 2.
        dim (int, optional): 1 => time masking, 2 => frequency masking. Defaults to 1.
        max_ratio (float | None, optional): For time masking only: additionally cap mask_len
            by int(max_ratio * valid_frames). (SpecAugment uses p; LB uses p=1.0).
            Defaults to None.
        replace (str, optional): Must be "zeros". Defaults to "zeros".

    Returns:
        torch.Tensor: Masked spectrogram of shape [B, T, F].
    """

    def __init__(
        self,
        mask_len: int = 15,
        num_masks: int = 2,
        dim: int = 1,
        max_ratio: float | None = None,
        replace: str = "zeros",
    ) -> None:
        super().__init__()
        if replace != "zeros":
            raise ValueError('SpecAugment masking baseline should be zeros only (replace="zeros").')
        if dim not in (1, 2):
            raise ValueError("dim must be 1 (time) or 2 (freq).")
        self.mask_len = int(mask_len)
        self.num_masks = int(num_masks)
        self.dim = int(dim)
        self.max_ratio = max_ratio
        self.replace = replace

    def forward(self, spectrogram: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Apply masking.

        Args:
            spectrogram (torch.Tensor): Input spectrogram [B, T, F].
            lengths (torch.Tensor | None, optional): Relative [0,1] or absolute frame lengths [B].
                Used only for time masking to avoid touching padding. Defaults to None.

        Returns:
            torch.Tensor: Masked spectrogram [B, T, F].
        """
        if spectrogram.dim() != 3:
            raise ValueError(f"Expected [B, T, F], got {tuple(spectrogram.shape)}.")

        B, T, Fdim = spectrogram.shape
        out = spectrogram.clone()

        if self.mask_len <= 0 or self.num_masks <= 0:
            return out

        dev = spectrogram.device

        if self.dim == 1:
            valid_T = _to_frame_lengths(lengths, T, B, dev)

            for b in range(B):
                L = int(valid_T[b].item())
                if L <= 1:
                    continue

                max_len = min(self.mask_len, L)
                if self.max_ratio is not None:
                    max_len = min(max_len, int(self.max_ratio * L))
                if max_len <= 0:
                    continue

                for _ in range(self.num_masks):
                    ml = int(torch.randint(0, max_len + 1, (1,), device=dev).item())
                    if ml == 0 or (L - ml) <= 0:
                        continue
                    start = int(torch.randint(0, L - ml + 1, (1,), device=dev).item())
                    out[b, start : start + ml, :] = 0

        else:  # dim == 2, frequency masking
            max_len = min(self.mask_len, Fdim)
            if max_len <= 0:
                return out

            for b in range(B):
                for _ in range(self.num_masks):
                    ml = int(torch.randint(0, max_len + 1, (1,), device=dev).item())
                    if ml == 0 or (Fdim - ml) <= 0:
                        continue
                    start = int(torch.randint(0, Fdim - ml + 1, (1,), device=dev).item())
                    out[b, :, start : start + ml] = 0

        return out


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

        Per the SpecAugment paper (Park et al., 2019):
        1. Choose a random center point c uniformly from [W, tau - W)
           independently for each sample in the batch.
        2. Choose a warp distance d uniformly from [-W, W].
        3. The point c is warped to w = c + d.
        4. The left region [0, c) is interpolated to fill [0, w),
           and the right region [c, tau) fills [w, tau).

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

        B = spectrogram.shape[0]
        tau = spectrogram.shape[2]
        F = spectrogram.shape[3]

        if tau - window <= window:
            return spectrogram.view(*original_size)

        # Sample center c in [W, tau - W) and distance d in [-W, W]
        # independently per sample in the batch.
        c = torch.randint(window, tau - window, (B,), device=spectrogram.device)
        d = torch.randint(-window, window + 1, (B,), device=spectrogram.device)
        w = c + d  # warp target point per sample

        # Build output without in-place mutation
        out = torch.empty_like(spectrogram)

        for i in range(B):
            ci = c[i].item()
            wi = w[i].item()

            # Clamp wi to [1, tau - 1] so both left and right regions are non-empty
            wi = max(1, min(wi, tau - 1))

            # Interpolate left region [0, ci) -> [0, wi)
            left = torch.nn.functional.interpolate(
                spectrogram[i : i + 1, :, :ci, :],
                (wi, F),
                mode=self.warp_mode,
                align_corners=True,
            )

            # Interpolate right region [ci, tau) -> [wi, tau)
            right = torch.nn.functional.interpolate(
                spectrogram[i : i + 1, :, ci:, :],
                (tau - wi, F),
                mode=self.warp_mode,
                align_corners=True,
            )

            out[i : i + 1, :, :wi, :] = left
            out[i : i + 1, :, wi:, :] = right

        out = out.view(*original_size)

        # Transpose back if freq warping was applied.
        if self.dim == 2:
            out = out.transpose(1, 2)

        return out


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

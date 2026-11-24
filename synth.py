"""Synthetic clips for the self-test, so the pipeline can be checked with no data.

- "Real" clips: random notes (harmonic tones) plus hi-hat-like noise bursts,
  made directly at 44.1 kHz. No deconvolution anywhere.
- "AI" clips: a random latent sequence pushed through a stack of untrained
  transposed-convolution layers (zero-insertion + filter + bias + LeakyReLU),
  the decoder pattern the paper analyses. The weights are random on purpose:
  the paper's claim is that the artifact comes from the architecture alone.

This is a test of the mechanism, not of real-world accuracy. The "AI" clips
don't sound like music. Use real songs and Suno/Udio songs for that (see RESEARCH_LOG.md).
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import fftconvolve, resample_poly

SR = 44100

# Two decoder layouts. A matches Encodec's strides (8, 5, 4, 2), so it upsamples
# by 320: latent frames at 44100/320 = 137.8 Hz, which should give a comb with
# 137.8 Hz spacing. B upsamples by 512: latent 86.1 Hz. The self-test trains on
# A and then tests on B, which plays the role of an unseen model version.
DECODERS = {"A": (8, 5, 4, 2), "B": (4, 4, 4, 4, 2)}


def make_real_clip(rng, sr=SR, dur=12.0):
    n = int(sr * dur)
    out = np.zeros(n)
    for _ in range(int(dur * 3)):  # notes
        f0 = 55.0 * 2 ** rng.uniform(0, 4.5)  # about 55-1240 Hz
        i0 = int(rng.uniform(0, dur - 0.3) * sr)
        i1 = min(n, i0 + int(rng.uniform(0.3, 1.5) * sr))
        t = np.arange(i1 - i0) / sr
        env = np.exp(-t * rng.uniform(1.5, 6.0)) * (1 - np.exp(-t * 200))
        inharm = rng.uniform(0, 1e-4)  # piano-like stretched harmonics
        note = np.zeros_like(t)
        for h in range(1, 26):
            fh = f0 * h * np.sqrt(1 + inharm * h * h)
            if fh > 0.95 * sr / 2:
                break
            note += h ** -1.3 * np.exp(-t * h * 0.3) * np.sin(2 * np.pi * fh * t + rng.uniform(0, 2 * np.pi))
        out[i0:i1] += rng.uniform(0.1, 0.4) * env * note
    for _ in range(int(dur * 6)):  # hi-hat-like noise bursts
        i0 = int(rng.uniform(0, dur - 0.2) * sr)
        length = int(0.15 * sr)
        burst = np.diff(rng.standard_normal(length + 1))  # first difference = high-pass
        seg = slice(i0, i0 + length)
        out[seg] += rng.uniform(0.02, 0.08) * burst[: len(out[seg])] * np.exp(-np.arange(len(out[seg])) / sr * 30)
    out += 0.002 * rng.standard_normal(n)  # noise floor
    return 0.8 * out / np.max(np.abs(out))


def make_ai_clip(rng, strides=DECODERS["A"], sr=SR, dur=12.0, base_channels=32, bias_std=0.05):
    hop = int(np.prod(strides))
    n = int(sr * dur)
    n_latent = int(np.ceil(n / hop)) + 1
    chans = [max(1, base_channels >> i) for i in range(len(strides))] + [1]

    # Latent frames: slowly varying random features with a small offset.
    x = uniform_filter1d(rng.standard_normal((chans[0], n_latent)), 3, axis=1)
    x += rng.normal(0, 0.3, (chans[0], 1))

    for i, k in enumerate(strides):
        c_in, c_out = chans[i], chans[i + 1]
        up = np.zeros((c_in, x.shape[1] * k))
        up[:, ::k] = x                                   # zero insertion (the key step)
        kernels = rng.standard_normal((c_in, 2 * k)) / np.sqrt(2 * k)
        y = fftconvolve(up, kernels, mode="same", axes=1)  # per-channel filter
        w = rng.standard_normal((c_out, c_in)) / np.sqrt(c_in)
        y = w @ y + rng.normal(0, bias_std, (c_out, 1))  # mix channels, add bias
        if i < len(strides) - 1:
            y = np.where(y > 0, y, 0.2 * y)              # LeakyReLU
        x = y
    out = x[0, :n]
    out -= out.mean()
    return 0.8 * out / np.max(np.abs(out))


def speed_up(x, up=18, down=17):
    """Play back faster by up/down (18/17 is about +1 semitone, like a pitch shift).

    Every frequency, and so every artifact peak, moves up by the same factor.
    The paper leaves this kind of change to future work.
    """
    return resample_poly(x, down, up)

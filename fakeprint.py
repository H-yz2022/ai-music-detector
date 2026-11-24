"""Fakeprint features from "A Fourier Explanation of AI-music Artifacts"
(Afchar, Meseguer-Brocal, Akesbi, Hennequin, ISMIR 2025, arXiv:2506.19108).

The idea in one paragraph
-------------------------
Most music generators end with a decoder built from transposed convolutions
("deconvolutions"). A stride-k deconvolution is the same as inserting k-1
zeros between samples and then filtering. Inserting zeros makes k copies of the
spectrum, so the peak every signal has at 0 Hz (from biases and ReLUs) gets
copied to every multiple of the layer's input sample rate. Stack a few layers
and you get a comb of thin, evenly spaced peaks across the whole spectrum. The
peaks depend on the architecture, not the training data, so they show up in
every song a given model makes.

How we measure it (matches the authors' compute_fakeprints.py)
--------------------------------------------------------------
1. Power STFT with n_fft = 16384 (about 2.7 Hz bins at 44.1 kHz), hop = 8192,
   Hann window, in dB, averaged over all frames and channels.
2. Keep 5-16 kHz, where music is mostly noise-like and the comb stands out.
3. Estimate the spectrum's lower envelope ("lower hull") from minima in sliding
   10-bin windows, subtract it, then clip and normalise to [0, 1].
   What's left is the fakeprint: mostly zero for real music, a comb for AI.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import make_interp_spline
from scipy.ndimage import uniform_filter1d
from scipy.signal import get_window

N_FFT = 1 << 14
HOP = N_FFT // 2
FMIN, FMAX = 5000.0, 16000.0
BUMP_HZ = 15.0  # bump removal width for the comb score, chosen on the dev sample (data\)


def average_spectrum_db(audio, sr, n_fft=N_FFT, hop=HOP, block=64):
    """Mean dB power spectrum over all STFT frames and channels.

    audio: (channels, samples). Returns (freqs, spectrum_db), length n_fft//2+1.
    Uses the same conventions as torchaudio.Spectrogram(n_fft, power=2):
    periodic Hann window, centred frames with reflect padding, no normalisation.
    """
    window = get_window("hann", n_fft)  # periodic, like torch.hann_window
    pad = n_fft // 2
    total = np.zeros(n_fft // 2 + 1)
    count = 0
    for ch in np.atleast_2d(audio):
        if len(ch) <= pad:  # reflect padding needs more samples than pad
            ch = np.pad(ch, (0, pad + 1 - len(ch)))
        x = np.pad(ch, pad, mode="reflect")
        frames = sliding_window_view(x, n_fft)[::hop]
        for start in range(0, len(frames), block):  # chunked to save memory
            power = np.abs(np.fft.rfft(frames[start:start + block] * window, axis=1)) ** 2
            db = 10.0 * np.log10(np.clip(power, 1e-10, 1e6))
            total += db.sum(axis=0)
            count += len(db)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    return freqs, total / count


def lower_hull(c, area=10):
    """Smooth lower envelope through the minima of sliding `area`-bin windows."""
    n = len(c)
    windows = sliding_window_view(c, area)
    idx = np.unique(windows.argmin(axis=1) + np.arange(len(windows)))
    if idx[0] != 0:
        idx = np.r_[0, idx]
    if idx[-1] != n - 1:
        idx = np.r_[idx, n - 1]
    return make_interp_spline(idx, c[idx], k=2)(np.arange(n))


def fakeprint(freqs, spectrum_db, fmin=FMIN, fmax=FMAX, area=10, min_db=-45.0, max_db=5.0):
    """Return (band_freqs, fakeprint in [0, 1])."""
    band = (freqs > fmin) & (freqs < fmax)
    f, c = freqs[band], spectrum_db[band]
    hull = np.clip(lower_hull(c, area), min_db, None)
    fp = np.clip(c - hull, 0.0, max_db)
    return f, fp / (1e-6 + fp.max())


def remove_bumps(f, fp, width_hz=BUMP_HZ):
    """Keep only thin peaks: subtract a running mean `width_hz` wide and drop what falls below it.

    Broad bumps (wider than about width_hz) line up with themselves at many
    shifts and give real songs false comb scores. Thin comb peaks survive.
    """
    k = max(3, int(round(width_hz / (f[1] - f[0]))) | 1)  # odd number of bins
    return np.clip(fp - uniform_filter1d(fp, k, mode="nearest"), 0.0, None)


def comb_score(f, fp, min_spacing=20.0, max_spacing=2000.0, bump_hz=0.0):
    """Extra signal, not from the paper: how strongly the fakeprint repeats.

    Slides the fakeprint against itself (autocorrelation). If it lines up
    again after a shift of N Hz, the peaks repeat every N Hz. Returns
    (score, spacing_hz). The score is the correlation at the best shift:
    near 0 means no regular peaks (real music), and near 1 means a clean comb.
    The spacing is roughly the decoder's latent frame rate.
    bump_hz > 0 first removes bumps wider than that (see remove_bumps).
    """
    df = f[1] - f[0]
    ac, lo, hi, peaks = _autocorr_peaks(f, fp, min_spacing, max_spacing, bump_hz)
    if len(peaks) == 0:
        return float(ac[lo:hi + 1].max()), float("nan")
    best = ac[peaks].max()
    # A comb repeating every s Hz also repeats every 2s, 3s, ... and those
    # multiples can score slightly higher, so take the smallest spacing that
    # is nearly as strong as the best one.
    k = int(peaks[np.argmax(ac[peaks] >= 0.8 * best)])
    return float(best), _refine(ac, k, df)


def _autocorr_peaks(f, fp, min_spacing, max_spacing, bump_hz):
    """Normalised autocorrelation of the fakeprint, the lag range, and its local peaks (in bins)."""
    df = f[1] - f[0]
    if bump_hz > 0:
        fp = remove_bumps(f, fp, bump_hz)
    x = fp - fp.mean()
    n = 1 << int(np.ceil(np.log2(2 * len(x))))
    spec = np.fft.rfft(x, n)
    ac = np.fft.irfft(spec * np.conj(spec), n)[: len(x)]
    ac /= ac[0] + 1e-12
    lo = max(1, int(np.ceil(min_spacing / df)))
    hi = min(len(ac) - 2, int(max_spacing / df))
    lags = np.arange(lo, hi + 1)
    return ac, lo, hi, lags[(ac[lags] > ac[lags - 1]) & (ac[lags] >= ac[lags + 1])]


def _refine(ac, k, df):
    """Parabolic refinement of an autocorrelation peak at lag k, in Hz."""
    a, b, c = ac[k - 1], ac[k], ac[k + 1]
    denom = a - 2 * b + c
    return float((k + (0.5 * (a - c) / denom if denom != 0 else 0.0)) * df)


def precise_comb(f, fp, top=5, min_spacing=20.0, max_spacing=2000.0, bump_hz=BUMP_HZ):
    """comb_sharpness at the `top` strongest autocorrelation peaks, not just one rough spacing.

    The comb score's single rough spacing can miss the comb (e.g. 183 Hz for a
    200 Hz comb, outside the +-3% search). Trying the few strongest repeat
    distances and keeping the sharpest fixes that. Returns (sharpness, exact spacing).
    """
    df = f[1] - f[0]
    x = remove_bumps(f, fp, bump_hz) if bump_hz > 0 else fp
    ac, _, _, peaks = _autocorr_peaks(f, x, min_spacing, max_spacing, 0.0)
    if len(peaks) == 0:
        return 0.0, float("nan")
    best = (0.0, float("nan"))
    for k in peaks[np.argsort(ac[peaks])[::-1][:top]]:
        best = max(best, comb_sharpness(f, x, _refine(ac, int(k), df), bump_hz=0.0), key=lambda r: r[0])
    return best


def comb_sharpness(f, fp, spacing, n=2000, bump_hz=BUMP_HZ):
    """How *exactly* the peaks repeat, near a rough spacing (e.g. from comb_score).

    A decoder repeats at one exact rate (e.g. 44100 / 2048 = 21.533 Hz), so only
    that rate lines up all of its peaks at once: a spacing 0.01 Hz off puts the
    300th peak 3 Hz away. For every candidate spacing near `spacing`, this
    averages the (bump-removed) fakeprint at all its multiples (a "harmonic
    sum"). A real comb gives one sharp spike; broad or irregular structure gives
    a smooth curve. Returns (sharpness, exact spacing): the spike's height as a
    z-score against the other candidates, and where it is. The n candidates are
    spaced so the highest multiple moves 1/4 of a bin per step (about +-3% around
    `spacing`), so every spacing gets the same number of candidates.
    """
    if not np.isfinite(spacing) or spacing <= 0:
        return 0.0, float("nan")
    x = remove_bumps(f, fp, bump_hz) if bump_hz > 0 else fp
    df = f[1] - f[0]
    k_top = max(1, int(f[-1] // spacing))
    step = 0.25 * df / k_top
    cands = spacing + (np.arange(n) - n // 2) * step
    cands = cands[cands > 2 * df]
    k = np.arange(1, k_top + 2)
    pos = cands[:, None] * k[None, :]
    inside = (pos >= f[0]) & (pos <= f[-1])
    h = np.where(inside, np.interp(pos, f, x), 0.0).sum(axis=1) / np.maximum(inside.sum(axis=1), 1)
    i = int(h.argmax())
    return float((h[i] - h.mean()) / (h.std() + 1e-12)), float(cands[i])


def extract(audio, sr, fmin=FMIN, fmax=FMAX):
    """All features for one clip, as a dict."""
    freqs, spec = average_spectrum_db(audio, sr)
    f, fp = fakeprint(freqs, spec, fmin, fmax)
    score, spacing = comb_score(f, fp)
    return {"spec_freqs": freqs, "spec_db": spec, "freqs": f, "fp": fp,
            "comb_score": score, "spacing_hz": spacing}

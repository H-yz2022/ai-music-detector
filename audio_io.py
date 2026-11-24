"""Load audio files as float arrays shaped (channels, samples).

WAV works out of the box (scipy). MP3/FLAC/OGG need the optional
`soundfile` package:  pip install soundfile
"""
import io
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif"}


def _read_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype.kind == "i":
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    elif data.dtype.kind == "u":  # 8-bit WAV is unsigned
        data = (data.astype(np.float64) - 128.0) / 128.0
    else:
        data = data.astype(np.float64)
    data = data[None, :] if data.ndim == 1 else data.T
    return data, int(sr)


def _read_with_soundfile(path):
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            f"Reading '{path.suffix}' files needs the soundfile package "
            "(pip install soundfile), or convert the file to WAV first."
        ) from exc
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    return data.T, int(sr)


def load_audio(path, target_sr=44100, max_duration=180.0):
    """Return (audio, sr) with audio shaped (channels, samples) at target_sr.

    Like the paper's code, long files are cut to `max_duration` seconds and
    everything is resampled to one rate (44.1 kHz) so that frequency bins line
    up across files. Proper resampling keeps every frequency where it was; it
    only removes content above the new Nyquist.
    """
    path = Path(path)
    if path.suffix.lower() == ".wav":
        data, sr = _read_wav(path)
    else:
        data, sr = _read_with_soundfile(path)
    return _trim_resample(data, sr, target_sr, max_duration)


def load_audio_bytes(blob, target_sr=44100, max_duration=180.0):
    """Like load_audio, for a file already in memory (e.g. read from a zip). Needs soundfile."""
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(blob), dtype="float64", always_2d=True)
    return _trim_resample(data.T, int(sr), target_sr, max_duration)


def _trim_resample(data, sr, target_sr, max_duration):
    if max_duration:
        data = data[:, : int(sr * max_duration)]
    if sr != target_sr:
        g = gcd(sr, int(target_sr))
        data = resample_poly(data, int(target_sr) // g, sr // g, axis=1)
    return data, int(target_sr)


def list_audio_files(folder):
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)

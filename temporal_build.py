r"""Idea 3: a feature for how each frequency changes over time, not just its average.

The fakeprint is the AVERAGE dB spectrum over the 30 s. Generators with no comb (Mureka, Lyria, Mubert) hide in
that average. So each song also gets, in the same two bands (1-8 kHz of 16 kHz audio, 5-16 kHz of 44.1 kHz audio,
after the same MP3 conversions as the fakeprints):
  tstd0 / tstd1   per frequency bin: standard deviation over time (STFT frames) of the dB level, minus its
                  31-bin running median, so a bin that moves more or less than its neighbours stands out
                  (a steady decoder peak moves less than the music around it); same bins as fp0 / fp1
  tco0 / tco1     coarse: per 100 Hz band from 0 Hz to Nyquist, the mean standard deviation over time and the
                  mean frame-to-frame change (2 x 80 values at 16 kHz, 2 x 220 at 44.1 kHz)
Two more, to test whether the fakeprint throws away other useful information (fpinfo_test.py):
  shape0 / shape1 mean dB per 100 Hz band, 0 Hz to Nyquist, after the same MP3 conversion (the overall shape of the
                  spectrum, which the fakeprint removes)
  stereo          side minus mid level in dB per 100 Hz band, 0-16 kHz, from the original file (the fakeprint is
                  made from a mono mix, so stereo is lost); -60 dB for mono files
The old fakeprints and precise combs are computed again in the same pass (fp0, fp1, sharp0, sharp1, ...), so both
feature sets come from exactly the same songs.

Songs (all streamed; no audio is kept, only features, in features_tm\):
  fma_small  the 8,000 FMA small clips (real), read from fma_small.zip on the server
  echoes     the Echoes songs (12 generators), only the start of each song read from Echoes.zip
  more44     the more44_test.py songs (Udio, Mureka, HeartMuLa, Sonauto, Lyria Pro 3), same picks
  more_real  the more_real.py songs (FMA large, netlabels, live concerts), same picks
  fresh44    the fresh44_test.py songs (new real songs, Suno V5.5, MiniMax, Lyria 3), same picks

  python -I temporal_build.py --sets fma_small --n 20 --out-dir <scratch>   # quick trial
  python -I temporal_build.py                                               # everything, ~21 GB streamed
Re-running skips what is already saved (fma_small and echoes are saved in parts of 1,000 songs).
"""
import argparse
import io
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402
from scipy.signal import get_window  # noqa: E402

import detect  # noqa: E402
import fakeprint as fpm  # noqa: E402
import fresh44_test as f44  # noqa: E402

HERE = Path(__file__).resolve().parent
FMA_SMALL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
ECHOES = "https://huggingface.co/datasets/Octavian97/Echoes/resolve/main/Echoes.zip"
AUDIO = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
SETS = ["fma_small", "echoes", "more44", "more_real", "fresh44"]
PART = 1000
MED_BINS = 31


def frames_db(audio, sr, n_fft=fpm.N_FFT, hop=fpm.HOP):
    """dB power per STFT frame, the same STFT as fakeprint.average_spectrum_db: (freqs, list of (frames, bins)
    arrays, one per channel)."""
    window = get_window("hann", n_fft)
    pad = n_fft // 2
    out = []
    for ch in np.atleast_2d(audio):
        if len(ch) <= pad:
            ch = np.pad(ch, (0, pad + 1 - len(ch)))
        frames = sliding_window_view(np.pad(ch, pad, mode="reflect"), n_fft)[::hop]
        out.append(np.vstack([10.0 * np.log10(np.clip(np.abs(np.fft.rfft(frames[s:s + 64] * window, axis=1)) ** 2,
                                                       1e-10, 1e6)) for s in range(0, len(frames), 64)]))
    return np.linspace(0, sr / 2, n_fft // 2 + 1), out


def band_features(data, src_sr, b):
    """Average spectrum (as detect.formatted_spectrum) plus the time features of one band."""
    from audio_io import load_audio_bytes
    from fetch_data import to_format
    audio, sr = load_audio_bytes(to_format(data, src_sr, b["sr"], b["mp3_level"]), target_sr=b["sr"],
                                 max_duration=f44.SECONDS)
    freqs, chans = frames_db(audio, sr)
    spec = np.concatenate(chans).mean(axis=0)  # mean over all frames and channels, as average_spectrum_db
    d = np.mean(chans, axis=0)  # channels averaged, per frame
    std = d.std(axis=0)
    flux = np.abs(np.diff(d, axis=0)).mean(axis=0) if len(d) > 1 else np.zeros_like(std)
    band = (freqs > b["fmin"]) & (freqs < b["fmax"])  # the fakeprint's bins
    wide = (freqs >= b["fmin"] - 200) & (freqs <= b["fmax"] + 200)  # a margin, so the running median has context
    detr = std[wide] - median_filter(std[wide], MED_BINS, mode="nearest")
    tstd = detr[band[wide]]
    lo = np.arange(0, sr / 2, 100.0)
    idx = np.minimum((freqs // 100).astype(int), len(lo) - 1)
    cnt = np.bincount(idx, minlength=len(lo))
    tco = np.concatenate([np.bincount(idx, std, len(lo)) / cnt, np.bincount(idx, flux, len(lo)) / cnt])
    shape = np.bincount(idx, spec, len(lo)) / cnt  # mean dB per 100 Hz band: the spectrum's overall shape
    return freqs, spec, tstd, tco, shape


def stereo_width(data, sr, top=16000.0, n_fft=4096):
    """Side minus mid level in dB per 100 Hz band, 0-16 kHz, from the original file (before the mono MP3 step).
    Mono files give -60 dB everywhere."""
    lo = np.arange(0, top, 100.0)
    if data.shape[1] < 2:
        return np.full(len(lo), -60.0)
    mid, side = (data[:, 0] + data[:, 1]) / 2, (data[:, 0] - data[:, 1]) / 2
    _, (dm,) = frames_db(mid, sr, n_fft, n_fft // 2)
    freqs, (ds,) = frames_db(side, sr, n_fft, n_fft // 2)
    pm, ps = (10 ** (d / 10).mean(0) for d in (dm, ds))  # mean power per bin
    ok = freqs < top
    idx = (freqs[ok] // 100).astype(int)
    cnt = np.maximum(np.bincount(idx, minlength=len(lo)), 1)
    m, s = np.bincount(idx, pm[ok], len(lo)) / cnt, np.bincount(idx, ps[ok], len(lo)) / cnt
    return np.clip(10 * np.log10((s + 1e-12) / (m + 1e-12)), -60.0, 20.0)


def song_features_tm(blob, bands):
    """fresh44_test.song_features plus the time features: (dict, "") or (None, reason)."""
    with sf.SoundFile(io.BytesIO(blob)) as snd:
        sr = snd.samplerate
        data = snd.read(int(sr * f44.SECONDS), dtype="float64", always_2d=True)
    if len(data) < sr * 10:
        return None, "shorter than 10 s"
    out = {"src_sr": sr, "stereo": stereo_width(data, sr).astype(np.float32)}
    for k, b in enumerate(bands):
        freqs, spec, tstd, tco, shape = band_features(data, sr, b)
        out[f"shape{k}"] = shape.astype(np.float32)
        f, fp = fpm.fakeprint(freqs, spec, b["fmin"], b["fmax"])
        if len(tstd) != len(fp):
            raise ValueError(f"band {k}: {len(tstd)} time bins vs {len(fp)} fakeprint bins")
        out[f"fp{k}"] = fp.astype(np.float32)
        out[f"sharp{k}"], out[f"spacing{k}"] = fpm.precise_comb(f, fp)
        out[f"tstd{k}"], out[f"tco{k}"] = tstd.astype(np.float32), tco.astype(np.float32)
    out["top"] = detect.content_top_khz(freqs, spec)
    return out, ""


def init_worker():
    """Worker start: hide decoder warnings, and make the other scripts' workers use song_features_tm."""
    f44.quiet_worker()
    import more44_test
    import more_real
    for m in (f44, more44_test, more_real):
        m.song_features = song_features_tm


def zip_start(url, offset, csize, method, name_len):
    """Only the start of one file of a remote zip (enough for 30 s) when it is stored uncompressed; else all of it."""
    if method != zipfile.ZIP_STORED:
        return f44.zip_member(url, offset, csize, method, name_len)
    head, _ = f44.http_get(url, offset, offset + 30 + name_len + 1023)
    if head[:4] != b"PK\x03\x04":
        raise ValueError("bad zip entry")
    n, e = int.from_bytes(head[26:28], "little"), int.from_bytes(head[28:30], "little")
    start = offset + 30 + n + e
    got = [len(head)]

    def read(a, b):
        data = f44.http_get(url, start + a, start + b)[0]
        got[0] += len(data)
        return data
    blob = f44.media_start(read, csize)
    return blob, got[0]


def zip_work(job):
    source, name, where, bands = job
    try:
        blob, got = (f44.zip_member if source == "real" else zip_start)(*where)
        out, err = song_features_tm(blob, bands)
        return source, name, out, err, got
    except Exception as exc:
        return source, name, None, f"{type(exc).__name__}: {exc}"[:150], 0


def zip_set(which, out_dir, bands, n, workers):
    """fma_small or echoes, in parts of PART songs: out_dir/<which>/part_XX.npz."""
    from build_fakeprints import echoes_name
    url = FMA_SMALL if which == "fma_small" else ECHOES
    big = f44.RemoteFile(url)
    entries = [i for i in zipfile.ZipFile(big).infolist() if not i.is_dir() and i.filename.lower().endswith(AUDIO)]
    entries = entries[:n] if n else entries
    total = big.fetched
    jobs = []
    for i in entries:
        if which == "fma_small":
            name, source = f"fma_{PurePosixPath(i.filename).stem}.mp3", "real"
        else:
            name = echoes_name(i.filename)
            source = PurePosixPath(name).parent.name.lower()
        jobs.append((source, name, (url, i.header_offset, i.compress_size, i.compress_type, len(i.filename.encode()))))
    folder = out_dir / which
    folder.mkdir(parents=True, exist_ok=True)
    print(f"{which}: {len(jobs)} songs in {(len(jobs) + PART - 1) // PART} parts")
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=init_worker) as pool:
        for p in range(0, len(jobs), PART):
            path = folder / f"part_{p // PART:02d}.npz"
            if path.exists():
                continue
            res = []
            for fut in as_completed([pool.submit(zip_work, j + (bands,)) for j in jobs[p:p + PART]]):
                source, name, out, err, got = fut.result()
                total += got
                if out is None:
                    print(f"  skip {name}: {err}")
                else:
                    res.append((source, name, out))
            if not res:
                sys.exit(f"{which}: no song in part {p // PART} could be read")
            res.sort(key=lambda r: r[1])
            np.savez_compressed(path, names=np.array([r[1] for r in res]), sources=np.array([r[0] for r in res]),
                                **{k: np.array([r[2][k] for r in res]) for k in res[0][2]})
            done = min(p + PART, len(jobs))
            print(f"  {which}: {done} / {len(jobs)} songs, {total / 1e9:.2f} GB read, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=SETS, choices=SETS)
    ap.add_argument("--out-dir", default=str(HERE / "features_tm"))
    ap.add_argument("--n", type=int, default=0, help="quick trial: only this many songs (fma_small, echoes)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bands = [{k: v for k, v in b.items() if k != "freqs"} for b in joblib.load(HERE / "model_twoband.joblib")["bands"]]

    import more44_test
    import more_real
    for m in (f44, more44_test, more_real):  # their process pools start workers with init_worker
        m.quiet_worker = init_worker
        m.song_features = song_features_tm
    t0 = time.time()
    for s in args.sets:
        print(f"\n===== {s} =====", flush=True)
        if s in ("fma_small", "echoes"):
            zip_set(s, out_dir, bands, args.n, args.workers)
        elif s == "more44":
            more44_test.download(SimpleNamespace(out_dir=str(out_dir / "more44"), n=0, seed=0,
                                                 workers=args.workers), bands)
        elif s == "more_real":
            if not (out_dir / "more_real" / "etree.npz").exists():
                sys.argv = ["more_real.py", "--out-dir", str(out_dir / "more_real"), "--workers", str(args.workers)]
                more_real.main()
        elif s == "fresh44":
            out = out_dir / "fresh44.npz"
            if not out.exists():
                f44.download(SimpleNamespace(out=str(out), n=0, seed=0, workers=args.workers), bands)
        print(f"  ({(time.time() - t0) / 60:.0f} min so far)", flush=True)
    print("\nDone. No audio kept.")


if __name__ == "__main__":
    main()

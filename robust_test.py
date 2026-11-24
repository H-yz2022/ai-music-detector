r"""Robustness test: do small, everyday changes to a song fool the detector?

Songs (only about the first 30 s of each is read from the server and kept in memory; no audio is saved):
- real: 70 FMA-large clips and 30 Jamendo songs, picked from fresh44 (never trained on);
- AI: 40 Suno V5.5 and 40 MiniMax Music 3.0 (44.1 kHz files) from fresh44, and 40 Udio songs from a
  blanchon/udio_dataset shard that more44_test.py did not use.
Each song is changed in 11 ways, and the clean original is kept as a baseline:
  speed -3 / -1 / +1 / +3 %   resampled, so pitch and tempo both change (like a record played faster)
  pitch -1 / +1 semitone      tempo kept (phase vocoder, then resampling)
  tempo +5 %                  pitch kept (phase vocoder)
  MP3 128k, Ogg 150k          re-encoded once more before predict's own conversion (Spotify streams Ogg Vorbis)
  EQ                          bass +6 dB below 200 Hz, treble -6 dB above 8 kHz
  noise 30 dB                 white noise, 30 dB below the song
A changed song that would clip is turned down to a 0.99 peak. Then every version gets the same two fakeprints
as predict (detect.formatted_spectrum). Scores: the default model (model_twoband.joblib) alone and with the
joint rule, the per-family model (model_twoband_family.joblib), and the exact comb. All cut-offs are the
saved ones; nothing is tuned here.

  python -I robust_test.py                            # ~200 MB read; results -> reports\robust_test.json
  python -I robust_test.py --n 2 --out <scratch dir>  # quick trial, 2 songs per source
Fakeprints of every version are kept (float16) in features_hq\unseen\robust.npz, so a later model can be
checked without reading the songs again (--rescore).
"""
import argparse
import io
import json
import os
import random
import sys
import time
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path, PurePosixPath

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import resample_poly, sosfilt  # noqa: E402

import detect  # noqa: E402
import fakeprint as fpm  # noqa: E402
from fresh44_test import (FMA_LARGE, HF_SOURCES, SECONDS, RemoteFile, hf_jobs, media_start, mp3_start,  # noqa: E402
                          quiet_worker, zip_member)
from more44_test import AUDIO, HF, _tar_header, hf_tree  # noqa: E402

HERE = Path(__file__).resolve().parent
PICK = {"real_fma": 70, "real_jamendo": 30, "suno_v5.5": 40, "minimax3": 40, "udio": 40}
USED_UDIO = {"0020.tar", "0132.tar", "0155.tar", "0183.tar", "0197.tar", "0207.tar", "0215.tar", "0244.tar",
             "0248.tar", "0261.tar"}  # the shards more44_test.py read
CONDITIONS = ["clean", "speed -3%", "speed -1%", "speed +1%", "speed +3%", "pitch -1 st", "pitch +1 st",
              "tempo +5%", "MP3 128k", "Ogg 150k", "EQ", "noise 30 dB"]


# ---------- the changes ----------

def resample(x, factor):
    """Resample by `factor` (output length = input length * factor), along time."""
    f = Fraction(factor).limit_denominator(1000)
    return resample_poly(x, f.numerator, f.denominator, axis=0)


def time_stretch(x, rate, n_fft=2048, hop=512):
    """Phase vocoder (as in librosa): rate > 1 is faster, pitch kept. x is 1-D."""
    win = np.hanning(n_fft + 1)[:-1]
    pad = np.pad(x, (n_fft // 2, n_fft // 2))
    n = 1 + (len(pad) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    S = np.fft.rfft(pad[idx] * win, axis=1)
    steps = np.arange(0, n - 1, rate)
    i, a = steps.astype(int), (steps % 1)[:, None]
    mag = (1 - a) * np.abs(S[i]) + a * np.abs(S[i + 1])
    adv = 2 * np.pi * hop * np.arange(S.shape[1]) / n_fft
    d = np.angle(S[i + 1]) - np.angle(S[i]) - adv
    d -= 2 * np.pi * np.round(d / (2 * np.pi))
    phase = np.angle(S[0]) + np.vstack([np.zeros(S.shape[1]), np.cumsum(adv + d, axis=0)[:-1]])
    frames = np.fft.irfft(mag * np.exp(1j * phase), n=n_fft, axis=1) * win
    out, norm = np.zeros(hop * (len(steps) - 1) + n_fft), np.zeros(hop * (len(steps) - 1) + n_fft)
    for k in range(len(steps)):  # overlap-add
        out[k * hop:k * hop + n_fft] += frames[k]
        norm[k * hop:k * hop + n_fft] += win ** 2
    out = out / np.maximum(norm, 1e-8)
    return out[n_fft // 2:n_fft // 2 + int(round(len(x) / rate))]


def shelf(f0, gain_db, sr, high):
    """RBJ cookbook shelving filter (slope 1) as second-order sections."""
    A, w = 10 ** (gain_db / 40), 2 * np.pi * f0 / sr
    al, c = np.sin(w) / 2 * np.sqrt(2), np.cos(w)
    s = -1 if high else 1  # the high shelf is the low shelf with the cosine terms flipped
    b = [A * ((A + 1) - s * (A - 1) * c + 2 * np.sqrt(A) * al), 2 * s * A * ((A - 1) - s * (A + 1) * c),
         A * ((A + 1) - s * (A - 1) * c - 2 * np.sqrt(A) * al)]
    a = [(A + 1) + s * (A - 1) * c + 2 * np.sqrt(A) * al, -2 * s * ((A - 1) + s * (A + 1) * c),
         (A + 1) + s * (A - 1) * c - 2 * np.sqrt(A) * al]
    return np.array([b + a]) / a[0]


def reencode(x, sr, fmt, level):
    buf = io.BytesIO()
    kw = {"subtype": "VORBIS"} if fmt == "OGG" else {"subtype": "MPEG_LAYER_III", "bitrate_mode": "CONSTANT"}
    x = np.clip(x, -1, 1)
    with sf.SoundFile(buf, "w", sr, x.shape[1], format=fmt, compression_level=level, **kw) as f:
        for i in range(0, len(x), 16384):  # small blocks: one big write overflows the Vorbis encoder's stack
            f.write(x[i:i + 16384])
    kbps = len(buf.getvalue()) * 8 / (len(x) / sr) / 1000
    y, _ = sf.read(io.BytesIO(buf.getvalue()), dtype="float64", always_2d=True)
    return y, kbps


def change(cond, x, sr, seed):
    """One changed version of x (samples x channels): (audio, kbps of the re-encode or None)."""
    kbps = None
    if cond == "clean":
        return x, None
    if cond.startswith("speed"):  # +3% = played 3% faster: shorter, everything 3% higher
        y = resample(x, 1 / (1 + float(cond.split()[1][:-1]) / 100))
    elif cond.startswith("pitch"):
        f = 2 ** (float(cond.split()[1]) / 12)
        y = resample(np.stack([time_stretch(c, 1 / f) for c in x.T], axis=1), 1 / f)
    elif cond.startswith("tempo"):
        y = np.stack([time_stretch(c, 1 + float(cond.split()[1][:-1]) / 100) for c in x.T], axis=1)
    elif cond == "MP3 128k":
        y, kbps = reencode(x, sr, "MP3", 0.65)
    elif cond == "Ogg 150k":
        y, kbps = reencode(x, sr, "OGG", 0.5)
    elif cond == "EQ":
        y = sosfilt(np.vstack([shelf(200, 6, sr, False), shelf(8000, -6, sr, True)]), x, axis=0)
    elif cond == "noise 30 dB":
        rms = np.sqrt(np.mean(x ** 2))
        y = x + np.random.default_rng(seed).standard_normal(x.shape) * rms * 10 ** (-30 / 20)
    else:
        raise ValueError(cond)
    peak = np.abs(y).max()
    return (y * 0.99 / peak if peak > 0.99 else y), kbps


# ---------- features of every version ----------

def features(x, sr, bands):
    out = {}
    for k, b in enumerate(bands):
        freqs, spec = detect.formatted_spectrum(x, sr, b["sr"], b["mp3_level"], SECONDS)
        f, fp = fpm.fakeprint(freqs, spec, b["fmin"], b["fmax"])
        out[f"fp{k}"] = fp.astype(np.float32)
        out[f"sharp{k}"], out[f"spacing{k}"] = fpm.precise_comb(f, fp)
    return out


def work(job):
    """Runs in a worker process: read the start of one song, change it every way, fakeprint each version."""
    source, name, where, bands = job
    try:
        if where[0] == "zip":
            blob, got = zip_member(*where[1:])
        elif where[0] == "http":
            blob, got = mp3_start(where[1])
        else:  # one member of a remote tar
            _, url, off, size = where
            rf = RemoteFile(url)
            blob = media_start(lambda a, b: rf.get(off + a, off + b), size)
            got = rf.fetched
        with sf.SoundFile(io.BytesIO(blob)) as snd:
            sr = snd.samplerate
            x = snd.read(int(sr * SECONDS), dtype="float64", always_2d=True)
        del blob
        if len(x) < sr * 10:
            return source, name, None, "shorter than 10 s", got
        seed = zlib.crc32(name.encode())
        res = {"src_sr": sr, "kbps": {}}
        for c in CONDITIONS:
            y, kbps = change(c, x, sr, seed)
            res[c] = features(y, sr, bands)
            if kbps:
                res["kbps"][c] = kbps
        return source, name, res, "", got
    except Exception as exc:  # a few files are broken
        return source, name, None, f"{type(exc).__name__}: {exc}", 0


# ---------- picking the songs ----------

def plan(n, seed):
    fresh = np.load(HERE / "features_hq" / "unseen" / "fresh44.npz")
    rng = random.Random(seed)
    want = {}
    for s in ("real_fma", "real_jamendo", "suno_v5.5", "minimax3"):
        names = sorted(str(m) for m, src, sr in zip(fresh["names"], fresh["sources"], fresh["src_sr"])
                       if src == s and sr >= 44100)
        want[s] = set(rng.sample(names, min(len(names), n or PICK[s])))
    jobs, fetched = [], 0
    big = RemoteFile(FMA_LARGE)
    for i in zipfile.ZipFile(big).infolist():
        name = f"fma_large/{PurePosixPath(i.filename).name}"
        if name in want["real_fma"]:
            jobs.append(("real_fma", name, ("zip", FMA_LARGE, i.header_offset, i.compress_size, i.compress_type,
                                            len(i.filename.encode()))))
    fetched += big.fetched
    for source, repo, folder, min_size, _ in HF_SOURCES:
        if source in want:
            jobs += [j for j in hf_jobs(source, repo, folder, min_size, 0, seed) if j[1] in want[source]]
    # Udio: the first songs of one shard more44_test.py did not use
    shards = sorted(x["path"] for x in hf_tree("blanchon/udio_dataset", "data") if x["path"].endswith(".tar"))
    path = rng.choice([p for p in shards if Path(p).name not in USED_UDIO])
    url = HF.format("blanchon/udio_dataset", path)
    rf, off, k = RemoteFile(url), 0, 0
    while k < (n or PICK["udio"]) and off + 512 <= rf.size:
        rf.seek(off)
        h = _tar_header(rf.read(512))
        if h is None:
            break
        name, size, typ = h
        if typ not in (b"x", b"g") and name.lower().endswith(AUDIO):
            jobs.append(("udio", f"udio/{Path(path).name}/{name}", ("tar", url, off + 512, size)))
            k += 1
        off += 512 + (size + 511) // 512 * 512
    fetched += rf.fetched
    print("Songs: " + ", ".join(f"{s} {sum(j[0] == s for j in jobs)}" for s in PICK) + f" (Udio from {path})")
    return jobs, fetched


def collect(args, bands):
    jobs, fetched = plan(args.n, args.seed)
    bands = [{k: v for k, v in b.items() if k != "freqs"} for b in bands]
    print(f"Reading and changing {len(jobs)} songs with {args.workers} workers ...")
    res, t0 = [], time.time()
    with ProcessPoolExecutor(args.workers, initializer=quiet_worker) as pool:
        futs = [pool.submit(work, j + (bands,)) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            source, name, out, err, got = fut.result()
            fetched += got
            if out is None:
                print(f"  skip {name}: {err}")
            else:
                res.append((source, name, out))
            if k % 20 == 0 or k == len(jobs):
                spent = time.time() - t0
                print(f"  {k} / {len(jobs)}, {fetched / 1e6:.0f} MB read, {spent / 60:.1f} min, "
                      f"about {spent / k * (len(jobs) - k) / 60:.0f} min left", flush=True)
    res.sort(key=lambda r: r[1])
    store = {"names": np.array([r[1] for r in res]), "sources": np.array([r[0] for r in res]),
             "src_sr": np.array([r[2]["src_sr"] for r in res]), "conditions": np.array(CONDITIONS),
             "read_mb": fetched / 1e6}
    for key in ("fp0", "fp1", "sharp0", "sharp1", "spacing0", "spacing1"):
        arr = np.array([[r[2][c][key] for r in res] for c in CONDITIONS])
        store[key] = arr.astype(np.float16) if key.startswith("fp") else arr
    for c in ("MP3 128k", "Ogg 150k"):
        store[f"kbps {c}"] = np.array([r[2]["kbps"][c] for r in res])
    Path(args.features).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.features, **store)
    print(f"Done: {fetched / 1e6:.0f} MB read in {(time.time() - t0) / 60:.1f} min, no audio kept. "
          f"Fakeprints -> {args.features}")
    full = {key: np.array([[r[2][c][key] for r in res] for c in CONDITIONS]) for key in ("fp0", "fp1")}
    return {**store, **full}  # score with the float32 fakeprints


# ---------- scores ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=0, help="songs per source, for a quick trial (0 = the full set)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default=None, help="folder for the results (default: reports\\ and features_hq\\unseen\\)")
    ap.add_argument("--rescore", action="store_true", help="score the saved fakeprints again, read no songs")
    ap.add_argument("--model", default=str(HERE / "model_twoband.joblib"))
    ap.add_argument("--family-model", default=str(HERE / "model_twoband_family.joblib"))
    args = ap.parse_args()
    out = Path(args.out) if args.out else None
    args.features = str((out or HERE / "features_hq" / "unseen") / "robust.npz")
    report = (out or HERE / "reports") / "robust_test.json"

    two, fam = joblib.load(args.model), joblib.load(args.family_model)
    if args.rescore:
        d = dict(np.load(args.features))
        d["fp0"], d["fp1"] = d["fp0"].astype(np.float32), d["fp1"].astype(np.float32)
    else:
        d = collect(args, two["bands"])
    src = d["sources"].astype(str)
    conds = [str(c) for c in d["conditions"]]
    groups = {"real": np.char.startswith(src, "real"), **{s: src == s for s in ("suno_v5.5", "minimax3", "udio")}}
    c16, c44 = two["comb_cutoffs"]
    results = {}
    for ci, c in enumerate(conds):
        X = np.hstack([d["fp0"][ci], d["fp1"][ci]])
        p = detect.ai_proba(two["model"], X)
        s16, s44 = d["sharp0"][ci], d["sharp1"][ci]
        flags = {"default": p > two["cutoff"],
                 "joint": detect.joint_score(two["real_scores"], p, s16, s44) > two["joint_cutoff"],
                 "per family": detect.ai_proba(fam["model"], X) > fam["cutoff"],
                 "comb": (s16 > c16) | (s44 > c44)}
        sp = np.where(s16 / c16 >= s44 / c44, d["spacing0"][ci], d["spacing1"][ci])
        results[c] = {m: {g: round(100 * float(f[k].mean()), 1) for g, k in groups.items()} for m, f in flags.items()}
        results[c]["spacing"] = {}
        for g in ("suno_v5.5", "minimax3"):  # does the exact spacing move with a speed or pitch change?
            k = groups[g] & flags["comb"]
            results[c]["spacing"][g] = round(float(np.median(sp[k])), 2) if k.any() else None
    print(f"\n{len(src)} songs: " + ", ".join(f"{g} {int(k.sum())}" for g, k in groups.items())
          + (f"; {float(d['read_mb']):.0f} MB read" if "read_mb" in d else ""))
    print(f"Cut-offs (saved): default P(AI) > {two['cutoff']:.3f}, joint {two['joint_cutoff']:.4f}, "
          f"per family P(AI) > {fam['cutoff']:.3f}, comb {c16:.2f} / {c44:.2f}")
    for c in ("MP3 128k", "Ogg 150k"):
        if f"kbps {c}" in d:
            print(f"  {c}: {np.mean(d[f'kbps {c}']):.0f} kbps on average")
    for m in ("default", "joint", "per family", "comb"):
        print(f"\n{m}: real songs flagged / AI songs caught (%)")
        print(f"{'':14s}{'real':>7s}{'Suno':>7s}{'MiniMax':>9s}{'Udio':>7s}")
        for c in conds:
            r = results[c][m]
            print(f"{c:14s}{r['real']:7.1f}{r['suno_v5.5']:7.1f}{r['minimax3']:9.1f}{r['udio']:7.1f}")
    print("\nMedian exact comb spacing (Hz) of songs with a comb:")
    for c in conds:
        print(f"  {c:14s} Suno {results[c]['spacing']['suno_v5.5']}   MiniMax {results[c]['spacing']['minimax3']}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"songs": {g: int(k.sum()) for g, k in groups.items()}, "results": results,
                                  "model": Path(args.model).name, "family_model": Path(args.family_model).name},
                                 indent=1))
    print(f"\nSaved {report}")


if __name__ == "__main__":
    main()

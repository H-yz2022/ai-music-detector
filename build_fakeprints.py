r"""Turn a big dataset into fakeprints only: download, compute, keep the features, delete the zip.

A fakeprint takes about 10 KB per song (compressed), so 5,000 songs take
about 50 MB instead of GBs of audio. The audio goes through the same steps as in
detect.py and fetch_data.py: 16 kHz, first 30 s, 1-8 kHz band. Real FMA clips
first get the same 40 kbps MP3 conversion as the real clips in data\.

  python -I build_fakeprints.py sonics --part 10 --exclude data_fresh\ai\sonics
  python -I build_fakeprints.py fma --exclude data_fresh\real\fma16k
  python -I build_fakeprints.py echoes

Output: features\sonics_part10.npz, features\fma_small.npz (the training pool),
and features\unseen\echoes.npz (12 other generators, kept out of the training pool).
Echoes songs get the same 40 kbps MP3 conversion as the real clips, so every
test song is in one format.
--exclude skips every song whose id appears in that folder (so the held-out
test songs, and their sister songs from the same prompt, are never trained on).
"""
import argparse
import io
import os
import re
import sys
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
# One math-library thread per worker: the workers already run in parallel, and
# each extra OpenBLAS thread reserves memory (it ran out of RAM without this).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import fakeprint as fpm  # noqa: E402
import fetch_data as fd  # noqa: E402
from audio_io import load_audio_bytes  # noqa: E402

HERE = Path(__file__).resolve().parent
FEATURES = HERE / "features"
# Echoes dataset (arXiv 2603.23667, CC BY-SA 4.0): AI songs from 12 generators.
ECHOES_URL = "https://huggingface.co/datasets/Octavian97/Echoes/resolve/main/Echoes.zip"


def song_key(name):
    """Song id: the first number in the file name (FMA track id, or SONICS prompt id)."""
    m = re.search(r"\d+", PurePosixPath(name).name)
    return m.group(0) if m else name


def generator_of(name, dataset):
    if dataset == "fma":
        return "real"
    if dataset == "echoes":  # <TYPE>/<PROVIDER>/<file>.mp3
        return PurePosixPath(name).parent.name.lower()
    m = re.search(r"_(suno|udio)_", name.lower())
    return m.group(1) if m else "ai"


def echoes_name(path):
    """'Echoes/TTA/acestep/x.mp3' -> 'TTA/acestep/x.mp3' (generation type / provider / file)."""
    parts = PurePosixPath(path).parts
    start = next((i for i, p in enumerate(parts) if p.upper() in ("TTA", "ATA")), max(0, len(parts) - 2))
    return "/".join(parts[start:])


def featurize(job):
    """Runs in a worker process. Returns (name, (freqs, fakeprint) or None, error).

    With `spectrum`, it keeps the average dB spectrum of the band instead of the fakeprint (so the
    fakeprint of any smaller band can be made later), plus a coarse 500 Hz-band spectrum of the whole
    range, to see how far up each source has content: (freqs, band spectrum, coarse spectrum).
    """
    name, blob, dataset, sr, fmin, fmax, max_duration, match_sr, level, spectrum = job
    try:
        if dataset == "fma":
            data, src_sr = sf.read(io.BytesIO(blob), dtype="float64", always_2d=True)
        elif dataset == "echoes":  # full songs: decode only the part we use, to save RAM
            with sf.SoundFile(io.BytesIO(blob)) as snd:
                src_sr = snd.samplerate
                data = snd.read(int(src_sr * max_duration) if max_duration else -1, dtype="float64", always_2d=True)
        if dataset in ("fma", "echoes"):
            if len(data) < src_sr * 10:  # same rule as fetch_data.py
                return name, None, "shorter than 10 s"
            blob = fd.to_format(data, src_sr, match_sr, level)
        audio, sr = load_audio_bytes(blob, target_sr=sr, max_duration=max_duration)
        freqs, spec = fpm.average_spectrum_db(audio, sr)
        if spectrum:
            band = (freqs > fmin) & (freqs < fmax)
            edges = np.arange(0, sr / 2 + 1, 500)
            coarse = [spec[(freqs >= a) & (freqs < b)].mean() for a, b in zip(edges[:-1], edges[1:])]
            return name, (freqs[band], spec[band].astype(np.float32), np.array(coarse, np.float32)), ""
        f, fp = fpm.fakeprint(freqs, spec, fmin, fmax)
        return name, (f, fp.astype(np.float16)), ""
    except Exception as exc:  # a few files are known to be corrupt
        return name, None, str(exc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["sonics", "fma", "echoes"])
    ap.add_argument("--part", type=int, default=10, help="SONICS part, 1-10")
    ap.add_argument("--zip", help="use a zip you already have (it is not deleted)")
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--exclude", action="append", default=[], help="skip song ids found in this folder (repeatable)")
    ap.add_argument("--n", type=int, default=0, help="only the first N songs, for a quick test (0 = all)")
    ap.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)),
                    help="parallel processes; each needs about 200-300 MB of RAM")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--fmin", type=float, default=1000.0)
    ap.add_argument("--fmax", type=float, default=8000.0)
    ap.add_argument("--max-duration", type=float, default=30.0)
    ap.add_argument("--match-sr", type=int, default=fd.MATCH_SR,
                    help="FMA/Echoes songs are first converted to MP3 at this rate (default 16000)")
    ap.add_argument("--mp3-level", type=float, default=fd.MATCH_MP3_LEVEL,
                    help="MP3 level for that conversion (0.8 = 40 kbps at 16 kHz; 0.4 = 192 kbps at 44.1 kHz)")
    ap.add_argument("--spectrum", action="store_true",
                    help="keep the band's average spectrum instead of the fakeprint (about 2x larger)")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.dataset == "sonics":
        url = fd.SONICS_URL.format(part=args.part)
        zpath, out = fd.DOWNLOADS / "sonics" / f"part_{args.part:02d}.zip", FEATURES / f"sonics_part{args.part:02d}.npz"
    elif args.dataset == "fma":
        url, zpath, out = fd.FMA_URL, fd.DOWNLOADS / "fma" / "fma_small.zip", FEATURES / "fma_small.npz"
    else:  # kept in a subfolder so it never joins the training pool (features\*.npz)
        url, zpath, out = ECHOES_URL, fd.DOWNLOADS / "echoes" / "Echoes.zip", FEATURES / "unseen" / "echoes.npz"
    zpath = Path(args.zip) if args.zip else zpath
    out = Path(args.out) if args.out else out
    skip = {song_key(p.name) for d in args.exclude for p in Path(d).iterdir()}

    if not args.zip:
        print(f"Downloading {url}\n  to {zpath}")
        fd.download(url, zpath)
        if args.dataset == "fma":
            print("Checking SHA1 ...")
            if fd.sha1_of(zpath) != fd.FMA_SHA1:
                sys.exit("Checksum mismatch: the download is incomplete or damaged. Delete it and run again.")

    names, fps, coarse, freqs, skipped = [], [], [], None, 0
    t0 = time.time()

    def collect(done):
        nonlocal freqs, skipped
        for fut in done:
            name, res, err = fut.result()
            if res is None:
                skipped += 1
                print(f"  skip {name}: {err}")
                continue
            freqs = res[0] if freqs is None else freqs
            names.append(name)
            fps.append(res[1])
            if len(res) > 2:
                coarse.append(res[2])

    with zipfile.ZipFile(zpath) as zf, ProcessPoolExecutor(args.workers) as pool:
        entries = [i for i in fd.audio_entries(zf) if song_key(i.filename) not in skip]
        entries = entries[: args.n] if args.n else entries
        print(f"{len(entries)} songs to process with {args.workers} workers ({len(skip)} song ids excluded)")
        pending = set()
        for k, info in enumerate(entries, 1):
            stem = PurePosixPath(info.filename)
            name = (f"fma_{stem.stem}.mp3" if args.dataset == "fma" else
                    echoes_name(info.filename) if args.dataset == "echoes" else stem.name)
            job = (name, zf.read(info), args.dataset, args.sr, args.fmin, args.fmax, args.max_duration,
                   args.match_sr, args.mp3_level, args.spectrum)
            pending.add(pool.submit(featurize, job))
            if len(pending) >= 4 * args.workers:  # keep only a few songs in memory at a time
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                collect(done)
            if k % 500 == 0:
                spent = time.time() - t0
                print(f"  {k} / {len(entries)} songs, {spent / 60:.1f} min, "
                      f"about {spent / k * (len(entries) - k) / 60:.0f} min left")
        collect(wait(pending)[0])

    if not names:
        sys.exit("No songs could be read.")
    order = np.argsort(names)
    names = np.array(names)[order]
    out.parent.mkdir(parents=True, exist_ok=True)
    extra = ({"spec": np.stack(fps)[order], "coarse": np.stack(coarse)[order],
              "coarse_edges": np.arange(0, args.sr / 2 + 1, 500)} if args.spectrum else {"fp": np.stack(fps)[order]})
    np.savez_compressed(out, freqs=freqs, names=names, gens=np.array([generator_of(n, args.dataset) for n in names]),
                        sr=args.sr, fmin=args.fmin, fmax=args.fmax, max_duration=args.max_duration, source=url,
                        match_sr=args.match_sr, mp3_level=args.mp3_level, **extra)
    counts = dict(zip(*np.unique([generator_of(n, args.dataset) for n in names], return_counts=True)))
    print(f"Saved {len(names)} fakeprints {counts} to {out} ({out.stat().st_size / 1e6:.1f} MB), "
          f"{skipped} skipped, {(time.time() - t0) / 60:.1f} min")

    if not args.zip and not args.keep_zip:
        gb = zpath.stat().st_size / 1e9
        zpath.unlink()
        print(f"Deleted {zpath.name} ({gb:.1f} GB).")


if __name__ == "__main__":
    main()

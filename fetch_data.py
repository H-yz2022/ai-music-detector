r"""Download a dataset, keep only a small sample, then delete the big download.

Big datasets are not kept on disk. This script downloads the zip, extracts
just the sample you ask for, and removes the zip. Run it again whenever you
need more or different songs.

  python -I fetch_data.py sonics --n 100     # 50 Suno + 50 Udio songs  -> data\ai\sonics
  python -I fetch_data.py fma --n 100        # 100 real clips (CC music) -> data\real\fma16k
  python -I fetch_data.py fma-artists        # track -> artist table for fma_small -> features\fma_artists.csv

Options:
  --part N      SONICS part to download, 1-10 (2.7-3.8 GB each; default 10, the smallest)
  --seed N      pick a different random sample
  --zip PATH    use a zip you already downloaded instead of downloading
  --keep-zip    don't delete the zip afterwards
  --add         add to the output folder instead of replacing what's there
  --out DIR     put the sample somewhere else (e.g. data_fresh\real\fma16k for a held-out test)
  --exclude DIR skip songs that already exist in DIR (so a fresh sample shares no songs)

Real songs are converted to the SONICS format (16 kHz mono MP3, 40 kbps) so the
detector compares AI vs real, not one file format vs another.
"""
import argparse
import hashlib
import io
import random
import re
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
DOWNLOADS = HERE.parent / "datasets"  # temporary; zips are deleted after use
SONICS_URL = "https://huggingface.co/datasets/awsaf49/sonics/resolve/main/fake_songs/part_{part:02d}.zip"
FMA_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_SHA1 = "ade154f733639d52e35e32f5593efe5be76c6d70"
FMA_META_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
FMA_META_SHA1 = "f0df49ffe5f2a6008d7dc83c6915b31835dfe733"
MATCH_SR = 16000
MATCH_MP3_LEVEL = 0.8  # soundfile constant-bitrate level 0.8 -> 40 kbps at 16 kHz (measured)
AUDIO = {".mp3", ".wav", ".flac", ".ogg"}


def download(url, dest):
    """Stream url to dest, resuming a partial file if one exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    try:
        resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers))
    except urllib.error.HTTPError as exc:
        if exc.code == 416:  # already complete
            return dest
        raise
    with resp:
        if have and resp.status != 206:  # server ignored the resume request
            have = 0
        total = have + int(resp.headers.get("Content-Length", 0))
        done = have
        next_pct = (have * 100 // total) // 10 * 10 + 10 if total else 0
        if have:
            print(f"  resuming at {have / 1e9:.2f} / {total / 1e9:.2f} GB")
        with open(dest, "ab" if have else "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and done * 100 // total >= next_pct:
                    print(f"  downloaded {done / 1e9:.2f} / {total / 1e9:.2f} GB")
                    next_pct += 10
    return dest


def sha1_of(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def audio_entries(zf):
    return [i for i in zf.infolist() if not i.is_dir() and PurePosixPath(i.filename).suffix.lower() in AUDIO]


def prepare_out(out, add):
    out.mkdir(parents=True, exist_ok=True)
    if not add:
        for p in out.iterdir():
            if p.suffix.lower() in AUDIO:
                p.unlink()


def sample_sonics(zpath, out, n, seed, exclude=frozenset()):
    """Copy a balanced random sample of Suno and Udio songs (already 16 kHz MP3)."""
    with zipfile.ZipFile(zpath) as zf:
        groups = {}
        for info in audio_entries(zf):
            if PurePosixPath(info.filename).name in exclude:
                continue
            m = re.search(r"_(suno|udio)_", info.filename.lower())
            groups.setdefault(m.group(1) if m else "other", []).append(info)
        rng = random.Random(seed)
        for tag in sorted(groups):
            pick = rng.sample(groups[tag], min(n // len(groups), len(groups[tag])))
            for info in pick:
                (out / PurePosixPath(info.filename).name).write_bytes(zf.read(info))
            print(f"  {tag}: {len(pick)} songs")


def to_match_format(data, sr):
    """Mono, 16 kHz, then written as 40 kbps MP3 bytes."""
    return to_format(data, sr)


def to_format(data, sr, out_sr=MATCH_SR, level=MATCH_MP3_LEVEL):
    """Mono, resampled to out_sr, then written as constant-bitrate MP3 bytes (level 0 = highest bitrate)."""
    mono = data.mean(axis=1)
    if sr != out_sr:
        from math import gcd
        g = gcd(sr, out_sr)
        mono = resample_poly(mono, out_sr // g, sr // g)
    mono = np.clip(mono, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, mono, out_sr, format="MP3", compression_level=level, bitrate_mode="CONSTANT")
    return buf.getvalue()


def sample_fma(zpath, out, n, seed, exclude=frozenset()):
    """Convert a random sample of FMA clips to the SONICS format."""
    with zipfile.ZipFile(zpath) as zf:
        entries = audio_entries(zf)
        random.Random(seed).shuffle(entries)
        kept = 0
        for info in entries:
            if kept >= n:
                break
            if f"fma_{PurePosixPath(info.filename).stem}.mp3" in exclude:
                continue
            try:
                data, sr = sf.read(io.BytesIO(zf.read(info)), dtype="float64", always_2d=True)
                if len(data) < sr * 10:  # skip broken or very short clips
                    continue
                (out / f"fma_{PurePosixPath(info.filename).stem}.mp3").write_bytes(to_match_format(data, sr))
                kept += 1
            except Exception as exc:  # a few FMA files are known to be corrupt
                print(f"  skip {info.filename}: {exc}")
        print(f"  real (FMA): {kept} clips converted to {MATCH_SR // 1000} kHz mono 40 kbps MP3")


def fma_artists(zpath, out):
    """Write track id -> artist id for the fma_small tracks (a small CSV) from the FMA metadata."""
    import pandas as pd
    with zipfile.ZipFile(zpath) as zf, zf.open("fma_metadata/tracks.csv") as f:
        tracks = pd.read_csv(f, index_col=0, header=[0, 1], low_memory=False)
    small = tracks[tracks[("set", "subset")] == "small"]
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"track_id": small.index, "artist_id": small[("artist", "id")]}).to_csv(out, index=False)
    print(f"  {len(small)} fma_small tracks by {small[('artist', 'id')].nunique()} artists -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["sonics", "fma", "fma-artists"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--part", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zip")
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--add", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--exclude")
    args = ap.parse_args()

    if args.dataset == "fma-artists":  # only a small track -> artist table is kept
        zpath = Path(args.zip) if args.zip else DOWNLOADS / "fma" / "fma_metadata.zip"
        if not args.zip:
            print(f"Downloading {FMA_META_URL}\n  to {zpath}")
            download(FMA_META_URL, zpath)
        if sha1_of(zpath) != FMA_META_SHA1:  # the zip's own CRC checks still catch damage while reading
            print("  warning: SHA1 differs from the published one (file may have been updated upstream)")
        fma_artists(zpath, Path(args.out) if args.out else HERE / "features" / "fma_artists.csv")
        if not args.zip and not args.keep_zip:
            zpath.unlink()
            print(f"Deleted {zpath.name}.")
        return

    if args.dataset == "sonics":
        url, zpath = SONICS_URL.format(part=args.part), DOWNLOADS / "sonics" / f"part_{args.part:02d}.zip"
        out = HERE / "data" / "ai" / "sonics"
    else:
        url, zpath = FMA_URL, DOWNLOADS / "fma" / "fma_small.zip"
        out = HERE / "data" / "real" / "fma16k"
    zpath = Path(args.zip) if args.zip else zpath
    out = Path(args.out) if args.out else out
    exclude = frozenset(p.name for p in Path(args.exclude).iterdir()) if args.exclude else frozenset()

    if not args.zip:
        print(f"Downloading {url}\n  to {zpath}")
        download(url, zpath)
    if args.dataset == "fma":
        print("Checking SHA1 ...")
        if sha1_of(zpath) != FMA_SHA1:
            sys.exit("Checksum mismatch: the download is incomplete or damaged. Delete it and run again.")

    prepare_out(out, args.add)
    (sample_sonics if args.dataset == "sonics" else sample_fma)(zpath, out, args.n, args.seed, exclude)
    size_mb = sum(p.stat().st_size for p in out.iterdir()) / 1e6
    print(f"Sample: {out}  ({size_mb:.0f} MB)")

    if not args.keep_zip:
        gb = zpath.stat().st_size / 1e9
        zpath.unlink()
        print(f"Deleted {zpath.name} ({gb:.1f} GB). Run this script again to re-download.")


if __name__ == "__main__":
    main()

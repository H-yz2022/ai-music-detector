r"""Test on new 44.1 kHz songs that were never trained on: two-band model vs the 16 kHz model.

Songs (only about the first 30 s of each is downloaded, straight from the server; no audio is saved):
- real, FMA large: 300 clips, one per artist, from artists not in fma_small (the training songs);
- real, Jamendo: the 79 songs of jam-alt (Creative Commons), a music source not in training;
- AI, Suno V5.5 (May 2026, CC BY 4.0): 150 songs. Training has Suno v5 (Echoes), and the 16 kHz
  model also has Suno v2-v3.5 (SONICS);
- AI, MiniMax Music 3.0 (CC BY 4.0): 150 songs, a generator not in training;
- AI, Lyria 3: 52 songs, a generator not in training (the dataset has no description, so the
  source is not verified).
Each song gets the same fakeprints as the training songs (detect.formatted_spectrum). The cut-offs
are the saved ones (model_twoband.joblib, model_family.joblib): nothing is tuned on these songs.
Features go to features_hq\unseen\fresh44.npz. Run again to re-score without downloading
(--refresh downloads again).

  python -I fresh44_test.py
  python -I fresh44_test.py --n 2 --out <scratch file>    # quick trial, 2 songs per source
"""
import argparse
import bz2
import csv
import io
import json
import os
import random
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import detect  # noqa: E402
import fakeprint as fpm  # noqa: E402

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "fakeprint-research-script"}
FMA_LARGE = "https://os.unil.cloud.switch.ch/fma/fma_large.zip"
FMA_META = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
# (source, Hugging Face dataset, folder, minimum file size, how many: 0 = all)
HF_SOURCES = [("real_jamendo", "jamendolyrics/jam-alt", "subsets", 0, 0),
              ("suno_v5.5", "Kukedlc/suno-ai-music-dataset", "audio", 1_000_000, 150),
              ("minimax3", "AngelSoftware/minimax-music-3-dataset", "audio/mp3", 1_000_000, 150),
              ("lyria3", "sleeping-ai/Lyria3", "", 0, 0)]
SECONDS = 30.0


def fetch(url, start=None, end=None, tries=6):
    """(bytes, response headers, final URL after redirects) of url, or of bytes start..end (inclusive).
    Retries network errors, and waits when the server says to slow down (HTTP 429)."""
    for k in range(tries):
        try:
            headers = dict(UA)
            if start is not None:
                headers["Range"] = f"bytes={start}-{'' if end is None else end}"
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                return r.read(), r.headers, r.geturl()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or k == tries - 1:
                raise
            wait = 2.0 ** k
            if exc.code == 429:
                try:
                    wait = min(300.0, float(exc.headers.get("Retry-After") or 60))
                except ValueError:
                    wait = 60.0
            time.sleep(wait)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def http_get(url, start=None, end=None, tries=6):
    """(bytes, Content-Range header) of url, or of bytes start..end (inclusive)."""
    data, headers, _ = fetch(url, start, end, tries)
    return data, headers.get("Content-Range")


class RemoteFile(io.RawIOBase):
    """A remote file that zipfile (or pyarrow) can read: each read fetches only the bytes it needs.
    Redirects are followed once (Hugging Face sends file reads on to a CDN); the CDN link is reused."""

    def __init__(self, url):
        super().__init__()
        self.url, self.pos, self.cache, self.fetched = url, 0, (0, b""), 0
        _, headers, self.direct = fetch(url, 0, 0)
        self.size = int(headers.get("Content-Range").split("/")[-1])

    def get(self, start, end):
        """Bytes start..end (inclusive). A CDN link expires after a while: then it is looked up again."""
        try:
            data, _ = http_get(self.direct, start, end)
        except urllib.error.HTTPError as exc:
            if exc.code not in (401, 403):
                raise
            _, _, self.direct = fetch(self.url, 0, 0)
            data, _ = http_get(self.direct, start, end)
        self.fetched += len(data)
        return data

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        c0, cb = self.cache
        if not (c0 <= self.pos and self.pos + n <= c0 + len(cb)):
            cb = self.get(self.pos, min(self.size, self.pos + max(n, 65536)) - 1)
            c0, self.cache = self.pos, (self.pos, cb)
        out = cb[self.pos - c0: self.pos - c0 + n]
        self.pos += len(out)
        return out


def zip_member(url, offset, csize, method, name_len):
    """One file of a remote zip, from its local header offset: (bytes, bytes downloaded)."""
    buf, _ = http_get(url, offset, offset + 30 + name_len + 512 + csize - 1)
    if buf[:4] != b"PK\x03\x04":
        raise ValueError("bad zip entry")
    n, e = struct.unpack("<HH", buf[26:30])
    start = 30 + n + e
    if start + csize > len(buf):
        more, _ = http_get(url, offset + len(buf), offset + start + csize - 1)
        buf += more
    data = buf[start:start + csize]
    if method == zipfile.ZIP_BZIP2:  # how FMA stores its clips
        data = bz2.decompress(data)
    elif method == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif method != zipfile.ZIP_STORED:
        raise ValueError(f"zip compression {method} not handled")
    return data, len(buf)


def decoded_seconds(blob):
    with sf.SoundFile(io.BytesIO(blob)) as snd:
        return len(snd.read(int(snd.samplerate * SECONDS), dtype="float32")) / snd.samplerate


def mp3_start(url):
    """The first bytes of a remote MP3, enough for SECONDS of audio: (bytes, bytes downloaded)."""
    blob, cr = http_get(url, 0, 640_000 - 1)
    total = int(cr.split("/")[-1]) if cr else len(blob)
    blob = media_start(lambda a, b: http_get(url, a, b)[0], total, blob)
    return blob, len(blob)


def media_start(read, total, blob=None):
    """Enough bytes from the start of an audio file (total bytes long) for SECONDS of audio.
    read(a, b) returns its bytes a..b; `blob` is what was already read from the start."""
    blob = read(0, min(total, 640_000) - 1) if blob is None else blob
    tag = 0
    if blob[:3] == b"ID3":  # tag (cover art etc.) before the audio
        s = blob[6:10]
        tag = 10 + ((s[0] << 21) | (s[1] << 14) | (s[2] << 7) | s[3])
    for _ in range(3):
        if len(blob) >= total:
            break
        try:
            got = decoded_seconds(blob)
        except Exception:
            got = 0.0
        if got >= SECONDS:
            break
        # bytes per second so far, plus a margin
        need = tag + (int((len(blob) - tag) / got * (SECONDS + 2) * 1.1) if got > 1 else len(blob) + 1_300_000)
        blob += read(len(blob), min(total, max(need, len(blob) + 65536)) - 1)
    return blob


def quiet_worker():
    """The MP3 decoder warns on stderr about every cut-off VBR file ('Xing stream size off'): hide that."""
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)


def song_features(blob, bands):
    """Both fakeprints, both precise combs and the content top of one song (the bytes of its audio file,
    or of its start): (dict, "") or (None, reason)."""
    with sf.SoundFile(io.BytesIO(blob)) as snd:
        sr = snd.samplerate
        data = snd.read(int(sr * SECONDS), dtype="float64", always_2d=True)
    if len(data) < sr * 10:  # same rule as build_fakeprints.py
        return None, "shorter than 10 s"
    out = {"src_sr": sr}
    for k, b in enumerate(bands):
        freqs, spec = detect.formatted_spectrum(data, sr, b["sr"], b["mp3_level"], SECONDS)
        f, fp = fpm.fakeprint(freqs, spec, b["fmin"], b["fmax"])
        out[f"fp{k}"] = fp.astype(np.float32)
        out[f"sharp{k}"], out[f"spacing{k}"] = fpm.precise_comb(f, fp)
    out["top"] = detect.content_top_khz(freqs, spec)  # of the last (44.1 kHz) band
    return out, ""


def featurize(job):
    """Runs in a worker process: download the start of one song, make both fakeprints."""
    source, name, where, bands = job
    try:
        blob, fetched = zip_member(*where[1:]) if where[0] == "zip" else mp3_start(where[1])
        out, err = song_features(blob, bands)
        return source, name, out, err, fetched
    except Exception as exc:  # a few files are broken
        return source, name, None, str(exc), 0


def fma_jobs(n, seed):
    """n FMA-large clips, one per artist, from artists with no song in fma_small."""
    meta = RemoteFile(FMA_META)
    raw = zipfile.ZipFile(meta).read("fma_metadata/raw_tracks.csv").decode("utf-8", errors="replace")
    csv.field_size_limit(1 << 30)
    artist = {}
    for row in csv.DictReader(io.StringIO(raw)):
        try:
            artist[int(row["track_id"])] = int(row["artist_id"])
        except (KeyError, TypeError, ValueError):
            pass
    small = {int(s[4:10]) for s in np.load(HERE / "features" / "fma_small.npz")["names"]}
    with open(HERE / "features" / "fma_artists.csv", newline="") as fh:
        small_artists = {int(r["artist_id"]) for r in csv.DictReader(fh)}
    small_artists |= {artist[t] for t in small if t in artist}
    big = RemoteFile(FMA_LARGE)
    entries = [i for i in zipfile.ZipFile(big).infolist() if i.filename.endswith(".mp3")]
    random.Random(seed).shuffle(entries)
    jobs, seen = [], set()
    for i in entries:
        t = int(PurePosixPath(i.filename).stem)
        a = artist.get(t)
        if t in small or a is None or a in small_artists or a in seen:
            continue
        seen.add(a)
        jobs.append(("real_fma", f"fma_large/{PurePosixPath(i.filename).name}",
                     ("zip", FMA_LARGE, i.header_offset, i.compress_size, i.compress_type, len(i.filename.encode()))))
        if len(jobs) >= n:
            break
    print(f"  FMA large: {len(entries)} clips, {len(small_artists)} training artists left out, {len(jobs)} picked")
    return jobs, meta.fetched + big.fetched


def hf_jobs(source, repo, folder, min_size, n, seed):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main{'/' + folder if folder else ''}?recursive=true"
    files = [x for x in json.loads(http_get(url)[0]) if x["type"] == "file" and x["path"].lower().endswith(".mp3")
             and x["size"] >= min_size and (source != "real_jamendo" or "/audio/" in x["path"])]
    files.sort(key=lambda x: x["path"])
    if n and len(files) > n:
        files = random.Random(seed).sample(files, n)
    print(f"  {source}: {len(files)} songs")
    base = f"https://huggingface.co/datasets/{repo}/resolve/main/"
    return [(source, f"{source}/{PurePosixPath(x['path']).name}", ("http", base + urllib.parse.quote(x["path"])))
            for x in files]


def download(args, bands):
    print("Picking songs ...")
    jobs, index_bytes = fma_jobs(args.n or 300, args.seed)
    for source, repo, folder, min_size, n in HF_SOURCES:
        jobs += hf_jobs(source, repo, folder, min_size, min(n, args.n) if args.n and n else (args.n or n), args.seed)
    bands = [{k: v for k, v in b.items() if k != "freqs"} for b in bands]
    print(f"Downloading the start of {len(jobs)} songs with {args.workers} workers ...")
    res, fetched, t0 = [], index_bytes, time.time()
    with ProcessPoolExecutor(args.workers, initializer=quiet_worker) as pool:
        futs = [pool.submit(featurize, j + (bands,)) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            source, name, out, err, got = fut.result()
            fetched += got
            if out is None:
                print(f"  skip {name}: {err}")
            else:
                res.append((source, name, out))
            if k % 50 == 0:
                spent = time.time() - t0
                print(f"  {k} / {len(jobs)}, {fetched / 1e6:.0f} MB downloaded, {spent / 60:.1f} min, "
                      f"about {spent / k * (len(jobs) - k) / 60:.0f} min left")
    res.sort(key=lambda r: r[1])
    out = {key: np.array([r[2][key] for r in res]) for key in res[0][2]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, names=np.array([r[1] for r in res]), sources=np.array([r[0] for r in res]),
                        downloaded_mb=fetched / 1e6, **out)
    print(f"Saved {len(res)} songs to {args.out}; {fetched / 1e6:.0f} MB downloaded, no audio kept.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "features_hq" / "unseen" / "fresh44.npz"))
    ap.add_argument("--n", type=int, default=0, help="songs per source, for a quick trial (0 = the full set)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--refresh", action="store_true", help="download again even if --out exists")
    ap.add_argument("--model", default=str(HERE / "model_twoband.joblib"), help="the two-band model to test")
    args = ap.parse_args()

    two = joblib.load(args.model)
    print(f"Two-band model: {Path(args.model).name}")
    fam = joblib.load(HERE / "model_family.joblib")
    if args.refresh or not Path(args.out).exists():
        download(args, two["bands"])
    d = np.load(args.out)
    src, fp16, fp44 = d["sources"], d["fp0"], d["fp1"]
    print(f"\n{len(src)} songs ({float(d['downloaded_mb']):.0f} MB were downloaded for them)")

    p2 = detect.ai_proba(two["model"], np.hstack([fp16, fp44]))
    flag2 = p2 > two["cutoff"]
    classes = [str(c) for c in fam["model"].classes_]
    proba = fam["model"].predict_proba(fp16)
    flag1 = 1 - proba[:, classes.index("real")] > fam["cutoff"]
    c16, c44 = two["comb_cutoffs"]
    comb = (d["sharp0"] > c16) | (d["sharp1"] > c44)
    spacing = np.where(d["sharp0"] / c16 >= d["sharp1"] / c44, d["spacing0"], d["spacing1"])
    hi = d["src_sr"] >= two["min_sr"]  # what detect.py predict sends to the two-band model
    flag = np.where(hi, flag2, flag1)  # predict's verdict
    if "joint_cutoff" in two:  # classifier or comb, one joint 5% cut-off (train_twoband.py)
        flagj = detect.joint_score(two["real_scores"], p2, d["sharp0"], d["sharp1"]) > two["joint_cutoff"]
    else:
        flagj = np.zeros(len(src), bool)
        print("(model_twoband.joblib has no joint cut-off yet: run train_twoband.py again)")

    print(f"\nCut-offs (saved, 5% of real training songs): two-band P(AI) > {two['cutoff']:.3f}, "
          f"16 kHz model P(AI) > {fam['cutoff']:.3f}, precise comb {c16:.2f} (1-8 kHz) / {c44:.2f} (5-16 kHz)")
    print(f"\n{'source':14s}{'songs':>6s}{'44.1k+':>8s}{'two-band':>10s}{'joint':>8s}{'16 kHz':>8s}{'predict':>9s}"
          f"{'comb':>7s}   most common comb spacing")
    for s in sorted(set(src), key=lambda s: (not s.startswith("real"), s)):
        m = src == s
        two_rate = (f"{100 * flag2[m & hi].mean():9.1f}%{100 * flagj[m & hi].mean():7.1f}%" if (m & hi).any()
                    else f"{'-':>10s}{'-':>8s}")
        exact = spacing[m & comb]
        sp = np.round(exact, 1)
        common = ""
        if len(sp):
            vals, counts = np.unique(sp, return_counts=True)
            top = float(np.median(exact[sp == vals[np.argmax(counts)]]))
            common = f"{top:.2f} Hz ({counts.max()} songs){detect.known_spacing(top)}"
        print(f"{s:14s}{m.sum():6d}{(m & hi).sum():8d}{two_rate}{100 * flag1[m].mean():7.1f}%"
              f"{100 * flag[m].mean():8.1f}%{100 * comb[m].mean():6.0f}%   {common}")
    print("(real rows: share wrongly flagged as AI; AI rows: share caught. 44.1k+: files at 44.1 kHz or more.\n"
          " two-band: on those files only. joint: two-band classifier or precise comb, one joint 5% cut-off.\n"
          " 16 kHz: model_family on every file. predict: two-band for 44.1k+ files, 16 kHz model for the rest,\n"
          " like detect.py predict. comb: an exact comb in either band.)")
    real = np.char.startswith(src.astype(str), "real")
    for tag, f in (("two-band", flag2), ("joint", flagj)):
        print(f"All real 44.1k+ songs flagged, {tag}: {int(f[real & hi].sum())} / {int((real & hi).sum())} "
              f"({100 * f[real & hi].mean():.1f}%)")
    rates = {s: sorted({int(r) for r in d["src_sr"][src == s]}) for s in sorted(set(src))}
    print("Sample rates: " + "; ".join(f"{s} {', '.join(f'{r / 1000:g}' for r in v)} kHz" for s, v in rates.items()))

    print("\nReal songs flagged by the two-band model, by how far up they have content:")
    for lo, up in ((0, 16), (16, 99)):
        m = real & hi & (d["top"] >= lo) & (d["top"] < up)
        if m.any():
            print(f"  {lo:2d}-{up if up < 99 else '':>2} kHz {100 * flag2[m].mean():5.1f}%  ({int(m.sum())} songs)")
    print("16 kHz model, AI songs caught: most like ...")
    for s in sorted(set(src[~real])):
        m = (src == s) & flag1
        if m.any():
            fams = [c for c in classes if c != "real"]
            sub = proba[m][:, [classes.index(c) for c in fams]]
            vals, counts = np.unique(np.array(fams)[sub.argmax(axis=1)], return_counts=True)
            print(f"  {s:12s} " + ", ".join(f"{v} {c}" for v, c in sorted(zip(vals, counts), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()

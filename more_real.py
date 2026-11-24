r"""More varied real songs for training (fakeprints only; no audio is kept).

All the real training songs so far are FMA small: 30 s clips from 8 genres of one archive. LightGBM
learned that look too closely (nonlinear_test.py: 38% of new real songs flagged). This adds:
- fma_large: 600 FMA-large clips, one per artist, from artists in neither fma_small nor fresh44
  (all 161 FMA genres, not just 8). The 300 fresh44 clips are skipped;
- netlabels: 450 songs from the Internet Archive's netlabels collection (Creative Commons), one per
  artist, 30 s from 40% into the song;
- etree: 450 songs from the Live Music Archive (live concerts, trade-friendly), one per band,
  30 s from 40% into the song.
Jamendo is left out on purpose, so fresh44's Jamendo songs stay a real source never trained on.
Downloads stop at --budget-gb. Each song gets the same fakeprints as the others
(fresh44_test.song_features). Output: features_hq\more_real\{fma_large,netlabels,etree}.npz,
and features_hq\more_real\fma_genres.json (genre of each FMA track used anywhere, for checks).

  python -I more_real.py --n 3 --out-dir <scratch folder>     # quick trial, 3 songs per source
  python -I more_real.py
"""
import argparse
import csv
import io
import json
import random
import re
import sys
import time
import urllib.parse
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from fresh44_test import (FMA_LARGE, FMA_META, RemoteFile, http_get, quiet_worker, song_features,  # noqa: E402
                          zip_member)

HERE = Path(__file__).resolve().parent
IA_SOURCES = {"netlabels": 'collection:netlabels AND mediatype:audio AND format:"VBR MP3"',
              "etree": 'collection:etree AND format:"VBR MP3"'}
OFFSET = 0.4  # where in the song the 30 s start (Internet Archive songs)
TITLE_ARTIST = set()  # sources whose items have no creator: take the artist from the title (newreal_test.py)


def fma_genre(s):
    m = re.search(r"'genre_title':\s*'([^']*)'", s or "")
    return m.group(1) if m else ""


def fma_part(n, seed):
    """n FMA-large clips: one per artist, artists in neither fma_small nor fresh44 (whose 300 clips came
    first in the same pick order). Returns (jobs, genres of all FMA tracks used, bytes read)."""
    meta = RemoteFile(FMA_META)
    raw = zipfile.ZipFile(meta).read("fma_metadata/raw_tracks.csv").decode("utf-8", errors="replace")
    csv.field_size_limit(1 << 30)
    artist, genre = {}, {}
    for row in csv.DictReader(io.StringIO(raw)):
        try:
            t = int(row["track_id"])
            artist[t] = int(row["artist_id"])
            genre[t] = fma_genre(row.get("track_genres"))
        except (KeyError, TypeError, ValueError):
            pass
    small = {int(s[4:10]) for s in np.load(HERE / "features" / "fma_small.npz")["names"]}
    with open(HERE / "features" / "fma_artists.csv", newline="") as fh:
        small_artists = {int(r["artist_id"]) for r in csv.DictReader(fh)}
    small_artists |= {artist[t] for t in small if t in artist}
    fresh = {int(PurePosixPath(s).stem) for s in np.load(HERE / "features_hq" / "unseen" / "fresh44.npz")["names"]
             if str(s).startswith("fma_large/")}
    fresh_artists = {artist[t] for t in fresh if t in artist}
    big = RemoteFile(FMA_LARGE)
    entries = [i for i in zipfile.ZipFile(big).infolist() if i.filename.endswith(".mp3")]
    random.Random(seed).shuffle(entries)
    jobs, seen = [], set()
    for i in entries:
        t = int(PurePosixPath(i.filename).stem)
        a = artist.get(t)
        if t in small or t in fresh or a is None or a in small_artists or a in fresh_artists or a in seen:
            continue
        seen.add(a)
        jobs.append(("fma_large", f"fma_large/{PurePosixPath(i.filename).name}", f"fma_artist{a}", genre.get(t, ""),
                     ("zip", FMA_LARGE, i.header_offset, i.compress_size, i.compress_type, len(i.filename.encode()))))
        if len(jobs) >= int(n * 1.3) + 2:  # spares: some FMA clips are broken (8 s long)
            break
    used = small | fresh | {int(PurePosixPath(j[1]).stem) for j in jobs}
    print(f"  fma_large: {len(jobs)} clips tried, n = {n} ({len(small_artists)} fma_small and {len(fresh_artists)} fresh44 "
          f"artists left out)")
    return jobs, {t: genre.get(t, "") for t in sorted(used)}, meta.fetched + big.fetched


def ia_part(source, n, seed, quick=False):
    """n Internet Archive items of a collection, one per artist (creator). Returns (jobs, bytes read).
    quick: only the first 10,000 items of the list (for a trial)."""
    items, cursor, got = [], None, 0
    while not (quick and items):
        q = {"q": IA_SOURCES[source], "fields": "identifier,creator,artist,title", "count": 10000}
        if cursor:
            q["cursor"] = cursor
        data, _ = http_get("https://archive.org/services/search/v1/scrape?" + urllib.parse.urlencode(q))
        got += len(data)
        r = json.loads(data)
        items += r.get("items", [])
        cursor = r.get("cursor")
        if not cursor:
            break
    by_artist = {}
    for it in items:
        c = it.get("creator")
        c = (c[0] if isinstance(c, list) and c else c) or ""
        if not c and source in TITLE_ARTIST:  # e.g. Jamendo albums: no creator, title "Artist - Album"
            c = it.get("artist") or str(it.get("title", "")).split(" - ")[0]
            c = (c[0] if isinstance(c, list) and c else c) or ""
        key = re.sub(r"\W+", " ", str(c)).strip().lower()
        if key:
            by_artist.setdefault(key, []).append(it["identifier"])
    rng = random.Random(seed)
    artists = sorted(by_artist)
    rng.shuffle(artists)
    jobs = [(source, it, f"{source}:{a}", "", ("ia", it)) for a in artists[: int(n * 1.3)]  # spare: some fail
            for it in [rng.choice(sorted(by_artist[a]))]]
    print(f"  {source}: {len(items)} items, {len(by_artist)} artists, {len(jobs)} tried (n = {n})")
    return jobs, got


BITRATES = {1: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
            2: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]}
RATES = {3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000]}


def mp3_sync(blob):
    """A piece cut from the middle of an MP3: start it at a frame header that is followed by another."""
    for i in range(min(len(blob) - 4, 500_000)):
        if blob[i] != 0xFF or blob[i + 1] & 0xE0 != 0xE0:
            continue
        ver, layer = (blob[i + 1] >> 3) & 3, (blob[i + 1] >> 1) & 3
        bri, sri, pad = blob[i + 2] >> 4, (blob[i + 2] >> 2) & 3, (blob[i + 2] >> 1) & 1
        if ver == 1 or layer != 1 or bri in (0, 15) or sri == 3:
            continue
        size = (144 if ver == 3 else 72) * BITRATES[1 if ver == 3 else 2][bri] * 1000 // RATES[ver][sri] + pad
        j = i + size
        if j + 1 < len(blob) and blob[j] == 0xFF and blob[j + 1] & 0xE0 == 0xE0:
            return blob[i:]
    raise ValueError("no MP3 frame found")


def seconds(x):
    try:
        parts = [float(p) for p in str(x).split(":")]
    except ValueError:
        return 0.0
    return sum(p * 60 ** k for k, p in enumerate(reversed(parts)))


def ia_song(identifier):
    """One song of an Internet Archive item: 30 s from 40% into it. (bytes, file name, bytes read)."""
    data, _ = http_get(f"https://archive.org/metadata/{urllib.parse.quote(identifier)}")
    files = []
    for f in json.loads(data).get("files", []):
        fmt, length = str(f.get("format", "")), seconds(f.get("length"))
        kbps = re.match(r"(\d+)Kbps MP3", fmt)
        if f["name"].lower().endswith(".mp3") and (fmt == "VBR MP3" or (kbps and int(kbps.group(1)) >= 128)) \
                and length >= 60 and int(f.get("size", 0)) > 0:
            files.append((f["name"], int(f["size"]), length))
    if not files:
        raise ValueError("no MP3 of 60 s or more")
    name, size, length = random.Random(identifier).choice(sorted(files))
    url = f"https://archive.org/download/{urllib.parse.quote(identifier)}/{urllib.parse.quote(name)}"
    start = int(size * OFFSET)
    blob, _ = http_get(url, start, min(size, start + int(size / length * 33 * 1.2)) - 1)
    return mp3_sync(blob), name, len(data) + len(blob)


def work(job):
    """Runs in a worker process: one song -> both fakeprints."""
    source, name, group, genre, where, bands = job
    try:
        if where[0] == "zip":
            blob, fetched = zip_member(*where[1:])
        else:
            blob, fname, fetched = ia_song(where[1])
            name = f"{source}/{where[1]}/{fname}"
        out, err = song_features(blob, bands)
        return source, name, group, genre, out, err, fetched
    except Exception as exc:
        return source, name, group, genre, None, str(exc)[:120], 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(HERE / "features_hq" / "more_real"))
    ap.add_argument("--n", type=int, default=0, help="songs per source, for a quick trial (0 = full set)")
    ap.add_argument("--fma", type=int, default=600)
    ap.add_argument("--ia", type=int, default=450, help="songs per Internet Archive collection")
    ap.add_argument("--budget-gb", type=float, default=1.8, help="stop downloading after this much")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    two = joblib.load(HERE / "model_twoband.joblib")
    bands = [{k: v for k, v in b.items() if k != "freqs"} for b in two["bands"]]
    budget = args.budget_gb * 1e9

    print("Picking songs ...")
    want = {"fma_large": args.n or args.fma} | {s: args.n or args.ia for s in IA_SOURCES}
    jobs, genres, fetched = fma_part(want["fma_large"], args.seed)
    (out_dir / "fma_genres.json").write_text(json.dumps({str(k): v for k, v in genres.items()}))
    for s in IA_SOURCES:
        j, got = ia_part(s, want[s], args.seed, quick=bool(args.n))
        jobs += j
        fetched += got
    print(f"  index and lists: {fetched / 1e6:.0f} MB")
    by_src = [[j for j in jobs if j[0] == s] for s in want]  # interleave, so the budget can't starve one source
    jobs = [lst[k] for k in range(max(map(len, by_src))) for lst in by_src if k < len(lst)]

    res = {s: [] for s in want}
    t0, done = time.time(), 0
    with ProcessPoolExecutor(args.workers, initializer=quiet_worker) as pool:
        futs = {pool.submit(work, j + (bands,)): j[0] for j in jobs}
        for fut in as_completed(futs):
            if fut.cancelled():
                continue
            source, name, group, genre, out, err, got = fut.result()
            fetched += got
            done += 1
            if out is None:
                print(f"  skip {name}: {err}")
            elif len(res[source]) < want[source]:
                res[source].append((name, group, genre, out))
                if len(res[source]) == want[source]:  # this source is full: drop its spare songs
                    for f, s in futs.items():
                        if s == source:
                            f.cancel()
            if done % 50 == 0:
                print(f"  {done} / {len(jobs)}, " + ", ".join(f"{s} {len(v)}" for s, v in res.items())
                      + f"; {fetched / 1e6:.0f} MB, {(time.time() - t0) / 60:.1f} min")
            if fetched > budget or all(len(res[s]) >= want[s] for s in want):
                if fetched > budget:
                    print(f"  budget of {args.budget_gb:g} GB reached: stopping")
                for f in futs:
                    f.cancel()
                break
        pool.shutdown(wait=True, cancel_futures=True)
    for source, rows in res.items():
        if not rows:
            continue
        rows.sort(key=lambda r: r[0])
        feats = {k: np.array([r[3][k] for r in rows]) for k in rows[0][3]}
        np.savez_compressed(out_dir / f"{source}.npz", names=np.array([r[0] for r in rows]),
                            sources=np.array(["real"] * len(rows)), origin=np.array([source] * len(rows)),
                            groups=np.array([r[1] for r in rows]), genre=np.array([r[2] for r in rows]), **feats)
        print(f"Saved {len(rows)} {source} songs; sample rates "
              + ", ".join(f"{r / 1000:g}" for r in sorted(set(feats['src_sr'].tolist()))) + " kHz")
    print(f"{fetched / 1e6:.0f} MB downloaded in {(time.time() - t0) / 60:.1f} min; no audio kept.")


if __name__ == "__main__":
    main()

r"""More 44.1 kHz AI songs: Udio, and four generators not in training (Mureka, HeartMuLa, Sonauto, Lyria Pro 3).

Only about the first 30 s of each song is downloaded, straight from the server; no audio is saved.
- Udio: 1,000 songs, 100 from the start of each of 10 random shards of blanchon/udio_dataset (CC0);
- Mureka v9: all 662 songs of homura23/MUSIC8K (CC BY 4.0); HeartMuLa: 500 of its 989;
- Sonauto: 500 songs, 50 from each of 10 random places in ai-music/ai-music-deduplicated (MIT);
- Lyria Pro 3: the 100 songs of mippia/HAIM (made through Google's API; CC BY-NC 4.0). They sit in
  one parquet file, which is read in full (about 400 MB).
Each song gets the same two fakeprints as the training songs. Features go to features_hq\more44\<source>.npz
(one file per source, so a re-run only downloads what is missing).

Phases:
  python -I more44_test.py                 # download what is missing, then 'before'
  python -I more44_test.py --phase before  # the current model_twoband on the new songs (never trained on)
  python -I more44_test.py --phase after   # 70% of each new source into training, 30% held out; and
                                           # leave one new generator out
  python -I more44_test.py --phase after --family   # the same, with one boundary per generator family
Songs are split by group: for Udio and Sonauto a group is one shard or place (songs next to each other
may share a user), elsewhere one song. The split and every setting are fixed before any result is seen.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.parse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Download workers: one math thread each. The 'after' phase trains models in one process: 4 threads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4" if "after" in sys.argv else "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

import detect  # noqa: E402
from fresh44_test import RemoteFile, http_get, media_start, mp3_start, quiet_worker, song_features  # noqa: E402

HERE = Path(__file__).resolve().parent
HF = "https://huggingface.co/datasets/{}/resolve/main/{}"
AUDIO = (".mp3", ".ogg", ".flac", ".wav", ".m4a")
NEW = ["udio_new", "mureka_v9", "heartmula", "sonauto", "lyria_pro3"]


def hf_tree(repo, folder):
    """All entries of a Hugging Face dataset folder (following the listing's pages)."""
    import re
    import urllib.request
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main/{urllib.parse.quote(folder)}"
    out = []
    while url:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "fakeprint-research-script"}),
                                    timeout=60) as r:
            out += json.loads(r.read())
            m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link") or "")
            url = m.group(1) if m else None
    return out


# ---------- tar archives (WebDataset shards) ----------

def _tar_header(b):
    """(name, size, type) of a 512-byte tar header, or None if it is not a valid header."""
    if len(b) < 512 or b[257:262] != b"ustar":
        return None
    try:
        if int(b[148:156].strip(b"\0 ") or b"0", 8) != sum(b[:148]) + 32 * 8 + sum(b[156:512]):
            return None
        size = int(b[124:136].strip(b"\0 ") or b"0", 8)
    except ValueError:
        return None
    return b[:100].rstrip(b"\0").decode(errors="replace"), size, b[156:157]


def _find_header(rf, off):
    """Offset of the first tar header at or after `off` (found by its 'ustar' mark and checksum)."""
    off -= off % 512
    while off < rf.size:
        chunk = rf.get(off, min(rf.size, off + (1 << 20)) - 1)
        for i in range(0, len(chunk) - 511, 512):
            if _tar_header(chunk[i:i + 512]):
                return off + i
        off += len(chunk) - len(chunk) % 512
    raise ValueError("no tar header found")


def tar_songs(source, url, label, start, k, bands):
    """k songs from a remote tar, starting at byte `start`: walks the headers, reads only each song's start."""
    rf = RemoteFile(url)
    off = 0 if start == 0 else _find_header(rf, start)
    res, skips = [], []
    while len(res) < k and off + 512 <= rf.size:
        rf.seek(off)
        h = _tar_header(rf.read(512))
        if h is None:
            break  # end of the archive
        name, size, typ = h
        data = off + 512
        off = data + (size + 511) // 512 * 512
        if typ in (b"x", b"g") or not name.lower().endswith(AUDIO):
            continue  # extra header or metadata
        song = f"{source}/{label}/{name}"
        try:
            blob = media_start(lambda a, b: rf.get(data + a, data + b), size)
            out, err = song_features(blob, bands)
        except Exception as exc:  # a broken song
            out, err = None, str(exc)
        if out is None:
            skips.append((song, err))
        else:
            res.append((source, song, f"{source}:{label}", out))
    return res, skips, rf.fetched


# ---------- jobs ----------

def run_job(job):
    """Runs in a worker process. Returns (songs, skipped, bytes downloaded)."""
    kind, source, *rest, bands = job
    try:
        if kind == "tar":
            return tar_songs(source, *rest, bands)
        if kind == "http":
            name, url = rest
            blob, got = mp3_start(url)
            out, err = song_features(blob, bands)
            return ([(source, name, name, out)] if out else []), ([] if out else [(name, err)]), got
        if kind == "parquet":  # one row group of a parquet file with an 'audio' column
            import pyarrow.parquet as pq
            url, rg = rest
            rf = RemoteFile(url)
            t = pq.ParquetFile(rf).read_row_group(rg, columns=["audio", "track_id"])
            res, skips = [], []
            for audio, tid in zip(t.column("audio").to_pylist(), t.column("track_id").to_pylist()):
                name = f"{source}/{tid}"
                try:
                    out, err = song_features(audio["bytes"] if isinstance(audio, dict) else audio, bands)
                except Exception as exc:
                    out, err = None, str(exc)
                if out is None:
                    skips.append((name, err))
                else:
                    res.append((source, name, name, out))
            return res, skips, rf.fetched
    except Exception as exc:
        return [], [(f"{source} job {rest[:2]}", str(exc))], 0
    raise ValueError(kind)


def plan(n, seed):
    """Jobs per source. n > 0: a quick trial with n songs per source (no Lyria: it can only be read in full)."""
    rng = random.Random(seed)
    jobs = {s: [] for s in NEW}
    shards = sorted(x["path"] for x in hf_tree("blanchon/udio_dataset", "data") if x["path"].endswith(".tar"))
    for path in rng.sample(shards, 1 if n else 10):
        jobs["udio_new"].append(("tar", "udio_new", HF.format("blanchon/udio_dataset", path),
                                 Path(path).name, 0, n or 100))
    for folder, source, count in (("mureka_v9", "mureka_v9", 0), ("heartmula", "heartmula", 500)):
        files = sorted(x["path"] for x in hf_tree("homura23/MUSIC8K", folder) if x["path"].lower().endswith(".mp3"))
        files = rng.sample(files, n) if n else (rng.sample(files, count) if count else files)
        jobs[source] += [("http", source, f"{source}/{Path(p).name}", HF.format("homura23/MUSIC8K", urllib.parse.quote(p)))
                         for p in files]
    tars = [("sonauto/sonauto_000000.tar", 8), ("sonauto/sonauto_000001.tar", 2)]
    for path, places in tars[:1] if n else tars:
        url = HF.format("ai-music/ai-music-deduplicated", path)
        size = int(http_get(url, 0, 0)[1].split("/")[-1])
        for start in sorted(rng.randrange(size - 50_000_000) for _ in range(1 if n else places)):
            jobs["sonauto"].append(("tar", "sonauto", url, f"{Path(path).stem}@{start // 1_000_000}MB", start, n or 50))
    if not n:
        url = HF.format("mippia/HAIM", "A2_fake_audio/lyria_pro3-00000-of-00001.parquet")
        jobs["lyria_pro3"] = [("parquet", "lyria_pro3", url, rg) for rg in range(5)]
    return jobs


def download(args, bands):
    bands = [{k: v for k, v in b.items() if k != "freqs"} for b in bands]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = {s: j for s, j in plan(args.n, args.seed).items() if j and not (out_dir / f"{s}.npz").exists()}
    if not todo:
        return
    print("To download: " + ", ".join(f"{s} ({len(j)} jobs)" for s, j in todo.items()))
    t0, total = time.time(), 0
    with ProcessPoolExecutor(args.workers, initializer=quiet_worker) as pool:
        futs = {pool.submit(run_job, j + (bands,)): s for s, js in todo.items() for j in js}
        got = {s: [] for s in todo}
        left = {s: len(js) for s, js in todo.items()}
        for fut in as_completed(futs):
            s = futs[fut]
            res, skips, fetched = fut.result()
            total += fetched
            got[s] += res
            left[s] -= 1
            for name, err in skips[:3]:
                print(f"  skip {name}: {err}")
            if len(skips) > 3:
                print(f"  ... and {len(skips) - 3} more skipped")
            print(f"  {s}: +{len(res)} songs ({len(got[s])} so far, {left[s]} jobs left); "
                  f"{total / 1e6:.0f} MB downloaded, {(time.time() - t0) / 60:.1f} min")
            if left[s] == 0 and got[s]:  # this source is done: save it
                rows = sorted(got[s], key=lambda r: r[1])
                np.savez_compressed(out_dir / f"{s}.npz", names=np.array([r[1] for r in rows]),
                                    sources=np.array([r[0] for r in rows]), groups=np.array([r[2] for r in rows]),
                                    **{k: np.array([r[3][k] for r in rows]) for k in rows[0][3]})
                print(f"  saved {len(rows)} {s} songs")
    print(f"Done: {total / 1e6:.0f} MB downloaded in {(time.time() - t0) / 60:.1f} min, no audio kept.")


def load_new(out_dir):
    parts = [np.load(p) for p in sorted(Path(out_dir).glob("*.npz"))]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


def report(title, src, flags, comb=None, spacing=None):
    print(f"\n{title}")
    for s in [s for s in NEW if s in set(src)] + sorted(set(src) - set(NEW)):
        m = src == s
        line = f"  {s:12s}{m.sum():6d}" + "".join(f"{100 * f[m].mean():9.1f}%" for f in flags.values())
        if comb is not None:
            line += f"{100 * comb[m].mean():7.0f}%"
            sp = spacing[m & comb]
            if len(sp):
                vals, counts = np.unique(np.round(sp, 1), return_counts=True)
                top = float(np.median(sp[np.round(sp, 1) == vals[np.argmax(counts)]]))
                line += f"   {top:.2f} Hz ({counts.max()} songs){detect.known_spacing(top)}"
        print(line)


def phase_before(d, two, fam):
    src = d["sources"]
    p2 = detect.ai_proba(two["model"], np.hstack([d["fp0"], d["fp1"]]))
    js = detect.joint_score(two["real_scores"], p2, d["sharp0"], d["sharp1"])
    classes = [str(c) for c in fam["model"].classes_]
    p1 = 1 - fam["model"].predict_proba(d["fp0"])[:, classes.index("real")]
    c16, c44 = two["comb_cutoffs"]
    comb = (d["sharp0"] > c16) | (d["sharp1"] > c44)
    spacing = np.where(d["sharp0"] / c16 >= d["sharp1"] / c44, d["spacing0"], d["spacing1"])
    print("Sample rates: " + "; ".join(f"{s} " + ", ".join(f"{r / 1000:g}" for r in sorted(set(d['src_sr'][src == s])))
                                       + " kHz" for s in NEW if s in set(src)))
    report(f"BEFORE: current models, these songs never trained on (share caught)\n  {'source':12s}{'songs':>6s}"
           f"{'two-band':>10s}{'joint':>10s}{'16 kHz':>10s}{'comb':>7s}   most common comb spacing", src,
           {"two": p2 > two["cutoff"], "joint": js > two["joint_cutoff"], "16k": p1 > fam["cutoff"]}, comb, spacing)
    print("(two-band: model_twoband classifier; joint: classifier or comb, one joint cut-off; 16 kHz: model_family.)")


def load_old():
    """The 12,357 training songs of train_twoband.py: (X, labels, groups, s16, s44)."""
    from cutoff_test import artists_of
    from hq_test import load
    fh, Xh, gh, nh, _, _ = load([str(HERE / "features_hq" / "fma_small.npz"), str(HERE / "features_hq" / "echoes.npz")],
                                "spec")
    fl, Xl, gl, nl = load([str(HERE / "features" / "fma_small.npz"), str(HERE / "features" / "unseen" / "echoes.npz")],
                          "fp")
    common = sorted(set(nh) & set(nl))
    ih, il = {n: i for i, n in enumerate(nh)}, {n: i for i, n in enumerate(nl)}
    X = np.hstack([Xl[[il[n] for n in common]], Xh[[ih[n] for n in common]]])
    labels, names = gh[[ih[n] for n in common]], np.array(common)
    art = artists_of(names, labels, str(HERE / "features" / "fma_artists.csv"))
    groups = np.array([f"artist{a}" if a is not None else n for a, n in zip(art, names)], dtype=object)
    c = np.load(HERE / "features_hq" / "precise_cache.npz")
    if not np.array_equal(c["names"], names):
        sys.exit("precise_cache.npz does not match: run train_twoband.py first")
    return X, labels, groups, c["s16"], c["s44"]


def split_train(sources, groups, seed=0, share=0.7):
    """True for the songs that go into training: about 70% of each source's groups."""
    rng = np.random.default_rng(seed)
    train = np.zeros(len(sources), bool)
    for s in sorted(set(sources)):
        g = sorted(set(groups[sources == s]))
        pick = rng.permutation(g)[: int(round(share * len(g)))]
        train |= (sources == s) & np.isin(groups, pick)
    return train


FAMILY = {"udio_new": "udio", "mureka_v9": "mureka", "lyria_pro3": "lyria"}  # source -> generator family


def phase_after(new, two, fresh_path, family=False):
    """Before vs after adding 70% of the new songs to training, and leave one new generator out.
    family: one boundary per generator family (multi-class, AI score = 1 - P(real)) instead of one yes/no boundary."""
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    if family:
        from family_test import ai_score, fit_family
    else:
        from logo_test import fit as fit_binary
    print("Loading the training songs ...")
    Xo, lo, go, s16o, s44o = load_old()
    tr_new = split_train(new["sources"], new["groups"])
    X = np.vstack([Xo, np.hstack([new["fp0"], new["fp1"]])])
    del Xo
    labels = np.concatenate([lo, new["sources"]])
    groups = np.concatenate([go, new["groups"].astype(object)])
    s16, s44 = np.concatenate([s16o, new["sharp0"]]), np.concatenate([s44o, new["sharp1"]])
    real, y = labels == "real", labels != "real"
    fam = np.array([FAMILY.get(s, s) for s in labels])

    class Model:  # one interface for both: fit on rows `use`, score = P(AI)
        def __init__(self, use):
            self.m = fit_family(X, fam, use) if family else fit_binary(X, y, use)

        def predict_proba(self, Xs):
            p = ai_score(self.m, Xs) if family else self.m.predict_proba(Xs)[:, 1]
            return np.column_stack([1 - p, p])

    fit = lambda X_, y_, use: Model(use)  # noqa: E731
    print(f"Classifier: {'one boundary per generator family (' + str(len(set(fam))) + ' classes)' if family else 'one yes/no boundary'}")
    is_new = np.arange(len(y)) >= len(lo)
    pool = ~is_new | np.concatenate([np.zeros(len(lo), bool), tr_new])  # the training songs of 'after'
    test = is_new & ~pool
    f = np.load(fresh_path)
    Xf, fsrc = np.hstack([f["fp0"], f["fp1"]]), f["sources"]
    print(f"  {len(lo)} old songs + {int(tr_new.sum())} new songs in training; {int(test.sum())} new songs held out")

    print("Cross-validation on the 'after' training songs (5 folds) ...")
    idx = np.flatnonzero(pool)
    oof = np.full(len(y), np.nan)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0)
                                .split(idx, labels[idx], groups[idx]), 1):
        use = pool.copy()
        use[idx[te]] = False
        oof[idx[te]] = fit(X, y, use).predict_proba(X[idx[te]])[:, 1]
        print(f"  fold {k}")
    rp = real & pool
    cut = float(np.quantile(oof[rp], 0.95))
    refs = {"clf": np.sort(oof[rp]), "s16": np.sort(s16[rp]), "s44": np.sort(s44[rp])}
    joint_cut = float(np.quantile(detect.joint_score(refs, oof[rp], s16[rp], s44[rp]), 0.95))
    print("Final 'after' model ...")
    m = fit(X, y, pool)

    def scores(model, cutoff, refs_, jcut, Xs, a16, a44):
        p = model.predict_proba(Xs)[:, 1] if isinstance(model, Model) else detect.ai_proba(model, Xs)
        return p > cutoff, detect.joint_score(refs_, p, a16, a44) > jcut

    bA = scores(two["model"], two["cutoff"], two["real_scores"], two["joint_cutoff"], X[test], s16[test], s44[test])
    bB = scores(m, cut, refs, joint_cut, X[test], s16[test], s44[test])
    fA = scores(two["model"], two["cutoff"], two["real_scores"], two["joint_cutoff"], Xf, f["sharp0"], f["sharp1"])
    fB = scores(m, cut, refs, joint_cut, Xf, f["sharp0"], f["sharp1"])

    print("Leave one new generator out ...")
    real_idx = np.flatnonzero(real)
    fold = np.empty(len(real_idx), int)
    for k, (_, te) in enumerate(GroupKFold(len(NEW)).split(real_idx, groups=groups[real_idx])):
        fold[te] = k
    logo = {}
    for i, g in enumerate(NEW):
        cal = real_idx[fold == i]
        use = pool & (labels != g)
        use[cal] = False
        mg = fit(X, y, use)
        logo[g] = float(np.mean(mg.predict_proba(X[labels == g])[:, 1] > np.quantile(mg.predict_proba(X[cal])[:, 1], 0.95)))
        print(f"  left out {g}: {100 * logo[g]:.0f}%")

    lt = labels[test]
    print(f"\nNEW SONGS HELD OUT (share caught). 'before' = current model, 'after' = trained with 70% of the new songs."
          f"\n  {'source':12s}{'test':>6s}{'before':>9s}{'after':>9s}{'joint before':>14s}{'joint after':>13s}"
          f"{'left out':>10s}")
    for g in NEW:
        mm = lt == g
        print(f"  {g:12s}{mm.sum():6d}{100 * bA[0][mm].mean():8.1f}%{100 * bB[0][mm].mean():8.1f}%"
              f"{100 * bA[1][mm].mean():13.1f}%{100 * bB[1][mm].mean():12.1f}%{100 * logo[g]:9.1f}%")
    print("  ('left out': trained on the old songs + 70% of the OTHER new sources, never this one; all its songs.)")
    print(f"\nTHE 728 SONGS OF fresh44_test.py (real rows: share flagged; AI rows: share caught; 44.1 kHz+ files)"
          f"\n  {'source':14s}{'songs':>6s}{'before':>9s}{'after':>9s}{'joint before':>14s}{'joint after':>13s}")
    hi = f["src_sr"] >= two["min_sr"]
    for s in sorted(set(fsrc), key=lambda s: (not s.startswith("real"), s)):
        mm = (fsrc == s) & hi
        print(f"  {s:14s}{mm.sum():6d}{100 * fA[0][mm].mean():8.1f}%{100 * fB[0][mm].mean():8.1f}%"
              f"{100 * fA[1][mm].mean():13.1f}%{100 * fB[1][mm].mean():12.1f}%")
    rf = np.char.startswith(fsrc.astype(str), "real") & hi
    print(f"  {'ALL REAL':14s}{rf.sum():6d}{100 * fA[0][rf].mean():8.1f}%{100 * fB[0][rf].mean():8.1f}%"
          f"{100 * fA[1][rf].mean():13.1f}%{100 * fB[1][rf].mean():12.1f}%")
    print(f"\n'After' cut-offs: P(AI) > {cut:.3f}, joint > {joint_cut:.4f}. Cross-validated catch rate on the 'after' "
          f"training songs:")
    for g in sorted(set(labels[pool & y])):
        mm = pool & (labels == g)
        print(f"  {g:12s}{mm.sum():6d}{100 * np.mean(oof[mm] > cut):8.1f}%")
    print(f"  {'ALL AI':12s}{(pool & y).sum():6d}{100 * np.mean(oof[pool & y] > cut):8.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["download", "before", "after"], default=None)
    ap.add_argument("--out-dir", default=str(HERE / "features_hq" / "more44"))
    ap.add_argument("--n", type=int, default=0, help="quick trial: songs per source (0 = the full set)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--family", action="store_true", help="'after' phase: one boundary per generator family")
    args = ap.parse_args()

    two = joblib.load(HERE / "model_twoband.joblib")
    if args.phase in (None, "download"):
        download(args, two["bands"])
    if args.phase in (None, "before"):
        phase_before(load_new(args.out_dir), two, joblib.load(HERE / "model_family.joblib"))
    if args.phase == "after":
        phase_after(load_new(args.out_dir), two, HERE / "features_hq" / "unseen" / "fresh44.npz", args.family)


if __name__ == "__main__":
    main()

r"""Check whole-song scoring (detect.py predict) on a new test set, never used before.

wholesong_test.py found that averaging the whole song catches more AI songs, but needs its own cut-off. That cut-off
and the real-song ranks now come from 1,130 complete real songs (calibrate_whole.py). This checks them on new songs:
  real   ~600 complete real songs from the Internet Archive, artists never used here: Jamendo albums, Community
         Audio music and netlabels (added before 2022, so before AI song generators; one song per artist)
  ai     ~600 Suno V5.5 and MiniMax songs from the same Hugging Face datasets as fresh44, but songs not used there
Each song is scored the way predict does it, both ways, with the model in model_twoband.joblib:
  windows  the median over 30 s windows (up to 8), with the 30 s cut-off (predict before)
  whole    the average over the whole song (up to 8 min), with the whole-song cut-off (predict now)
Download: --build streams the files (at most 12 MB each, ~6 GB, ~1 hour); only scores and fakeprints are kept
(features_tm\wholecheck\).

  python -I wholecheck_test.py --build --n 2 --out-dir <scratch>     # quick trial
  python -I wholecheck_test.py --build
  python -I wholecheck_test.py                                       # reports\wholecheck_test.json
"""
import argparse
import io
import json
import random
import sys
import time
import urllib.parse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
NOT_USED = ' AND format:"VBR MP3" AND addeddate:[* TO 2021-12-31] AND NOT collection:etree'
REAL = {"jamendo": "collection:jamendo-albums" + NOT_USED,
        "community": "collection:opensource_audio AND subject:music AND NOT collection:netlabels" + NOT_USED,
        "netlabels": "collection:netlabels AND mediatype:audio" + NOT_USED}
AI = [("suno_v5.5", "Kukedlc/suno-ai-music-dataset", "audio", 1_000_000),
      ("minimax3", "AngelSoftware/minimax-music-3-dataset", "audio/mp3", 1_000_000)]
MAX_BYTES = 12_000_000
WHOLE_SECONDS = 480.0


def used_groups():
    """Artists already used anywhere (training or tests)."""
    used = set()
    for p in list((HERE / "features_tm" / "newreal").glob("*.npz")) + list((HERE / "features_tm" / "more_real").glob("*.npz")):
        used |= set(np.load(p, allow_pickle=True)["groups"].astype(str))
    return used


def real_jobs(n, seed):
    import more_real
    more_real.IA_SOURCES = REAL
    more_real.TITLE_ARTIST = {"jamendo"}
    used = used_groups()
    jobs = []
    for s in REAL:
        cand, _ = more_real.ia_part(s, 10 ** 7, seed)  # every artist, one item each, shuffled
        fresh = [j for j in cand if j[2] not in used][: int(n * 1.3)]
        print(f"    {s}: {len(cand)} artists, {len(cand) - len([j for j in cand if j[2] not in used])} already used, "
              f"{len(fresh)} tried")
        jobs += [("real", s, j[1], ("ia", j[1])) for j in fresh]
    return jobs


def ai_jobs(n, seed):
    import fresh44_test as f44
    used = set(np.load(HERE / "features_tm" / "fresh44.npz")["names"].astype(str))
    jobs = []
    for source, repo, folder, min_size in AI:
        url = f"https://huggingface.co/api/datasets/{repo}/tree/main/{folder}?recursive=true"
        files = sorted(x["path"] for x in json.loads(f44.http_get(url)[0]) if x["type"] == "file"
                       and x["path"].lower().endswith(".mp3") and x["size"] >= min_size)
        files = [x for x in files if f"{source}/{PurePosixPath(x).name}" not in used]
        pick = random.Random(seed).sample(files, min(int(n * 1.5) + 2, len(files)))  # spares: some are 24 kHz
        print(f"    {source}: {len(files)} songs not used in fresh44, {len(pick)} picked")
        base = f"https://huggingface.co/datasets/{repo}/resolve/main/"
        jobs += [("ai", source, f"{source}/{PurePosixPath(x).name}", ("http", base + urllib.parse.quote(x))) for x in pick]
    return jobs


def ia_url(identifier):
    """A random MP3 of 60 s or more of an Internet Archive item (as more_real.ia_song): (url, bytes read)."""
    import re
    import more_real
    from fresh44_test import http_get
    data, _ = http_get(f"https://archive.org/metadata/{urllib.parse.quote(identifier)}")
    files = []
    for f in json.loads(data).get("files", []):
        fmt, length = str(f.get("format", "")), more_real.seconds(f.get("length"))
        kbps = re.match(r"(\d+)Kbps MP3", fmt)
        if f["name"].lower().endswith(".mp3") and (fmt == "VBR MP3" or (kbps and int(kbps.group(1)) >= 128)) \
                and length >= 60 and int(f.get("size", 0)) > 0:
            files.append(f["name"])
    if not files:
        raise ValueError("no MP3 of 60 s or more")
    name = random.Random(identifier).choice(sorted(files))
    return f"https://archive.org/download/{urllib.parse.quote(identifier)}/{urllib.parse.quote(name)}", name, len(data)


def work(job):
    """Runs in a worker: one song, scored as predict does (windows and whole song)."""
    import joblib
    import soundfile as sf
    import detect
    from fresh44_test import http_get
    which, source, name, where = job
    try:
        got = 0
        if where[0] == "ia":
            url, fname, got = ia_url(where[1])
            name = f"{source}/{where[1]}/{fname}"
        else:
            url = where[1]
        blob, _ = http_get(url, 0, MAX_BYTES - 1)
        got += len(blob)
        with sf.SoundFile(io.BytesIO(blob)) as snd:
            sr = snd.samplerate
            data = snd.read(int(sr * WHOLE_SECONDS), dtype="float32", always_2d=True).astype(np.float64)
        if len(data) < sr * 60:
            return which, source, name, None, "shorter than 60 s", got
        bundle = joblib.load(HERE / "model_twoband.joblib")
        if sr < bundle["min_sr"]:
            return which, source, name, None, f"{sr} Hz", got
        w = int(sr * bundle["max_duration"])
        n = min(detect.MAX_WINDOWS, len(data) // w)
        res = [detect.two_band_window(data[k * w:(k + 1) * w], sr, bundle) for k in range(n)]
        ps = detect.ai_proba(bundle["model"], np.array([r[0] for r in res]))
        x, c16, c44, _ = detect.two_band_window(data, sr, dict(bundle, max_duration=WHOLE_SECONDS))
        out = {"src_sr": sr, "seconds": len(data) / sr, "win_p": float(np.median(ps)),
               "win_s16": float(np.median([r[1][0] for r in res])), "win_s44": float(np.median([r[2][0] for r in res])),
               "fp": x.astype(np.float32), "sharp0": c16[0], "sharp1": c44[0]}
        return which, source, name, out, "", got
    except Exception as exc:
        return which, source, name, None, str(exc)[:120], 0


def build(out_dir, n_real, n_ai, workers, seed=7):
    import fresh44_test as f44
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Picking songs ...")
    jobs = real_jobs(n_real, seed) + ai_jobs(n_ai, seed)
    want = {s: n_real for s in REAL} | {s: n_ai for s, *_ in AI}
    res, fetched, t0 = {s: [] for s in want}, 0, time.time()
    with ProcessPoolExecutor(workers, initializer=f44.quiet_worker) as pool:
        futs = {pool.submit(work, j): j[1] for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            if fut.cancelled():
                continue
            which, source, name, out, err, got = fut.result()
            fetched += got
            if out is None:
                print(f"  skip {name}: {err}")
            elif len(res[source]) < want[source]:
                res[source].append((which, name, out))
                if len(res[source]) == want[source]:
                    for f, s in futs.items():
                        if s == source:
                            f.cancel()
            if k % 50 == 0:
                print(f"  {k} / {len(jobs)}, " + ", ".join(f"{s} {len(v)}" for s, v in res.items())
                      + f"; {fetched / 1e9:.2f} GB, {(time.time() - t0) / 60:.1f} min", flush=True)
            if all(len(res[s]) >= want[s] for s in want):
                for f in futs:
                    f.cancel()
                break
        pool.shutdown(wait=True, cancel_futures=True)
    rows = sorted(((s, *r) for s, v in res.items() for r in v), key=lambda r: r[2])
    feats = {k: np.array([r[3][k] for r in rows]) for k in rows[0][3]}
    np.savez_compressed(out_dir / "wholecheck.npz", names=np.array([r[2] for r in rows]),
                        sources=np.array([r[0] for r in rows]), kind=np.array([r[1] for r in rows]), **feats)
    print("Saved " + ", ".join(f"{s} {len(v)}" for s, v in res.items())
          + f"; {fetched / 1e9:.2f} GB downloaded in {(time.time() - t0) / 60:.1f} min; no audio kept.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--n", type=int, default=0, help="songs per source, for a quick trial")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", default=str(HERE / "features_tm" / "wholecheck"))
    ap.add_argument("--out", default=str(HERE / "reports" / "wholecheck_test.json"))
    args = ap.parse_args()
    if args.build:
        build(Path(args.out_dir), args.n or 200, args.n or 300, args.workers)
        return

    import joblib
    from detect import LEVELS, ai_proba, joint_score
    b = joblib.load(HERE / "model_twoband.joblib")
    if "whole" not in b:
        sys.exit("model_twoband.joblib has no whole-song calibration: run calibrate_whole.py first")
    w = b["whole"]
    d = dict(np.load(Path(args.out_dir) / "wholecheck.npz"))
    src, kind = d["sources"].astype(str), d["kind"].astype(str)
    pct = lambda ref, x: np.searchsorted(ref, x, side="right") / len(ref)  # noqa: E731
    p_whole = ai_proba(b["model"], d["fp"])
    scores = {
        "windows": (d["win_p"] > b["cutoff"],
                    joint_score(b["real_scores"], d["win_p"], d["win_s16"], d["win_s44"]) > b["joint_cutoff"],
                    pct(b["real_scores"]["clf"], d["win_p"])),
        "whole": (p_whole > w["cutoff"],
                  joint_score(w["real_scores"], p_whole, d["sharp0"], d["sharp1"]) > w["joint_cutoff"],
                  pct(w["real_scores"]["clf"], p_whole))}
    rows = {s: src == s for s in dict.fromkeys(src)} | {"ALL REAL": kind == "real", "ALL AI": kind == "ai"}
    out = {v: {r: [float(sc[0][m].mean()), float(sc[1][m].mean()), int(m.sum())] for r, m in rows.items()}
           for v, sc in scores.items()}
    print("Flagged, classifier (with comb)      windows, 30 s cut-off      whole song, whole-song cut-off")
    for r in rows:
        a, c = out["windows"][r], out["whole"][r]
        print(f"  {r:12s} ({a[2]:4d})   {100 * a[0]:6.1f}% ({100 * a[1]:5.1f}%)          {100 * c[0]:6.1f}% ({100 * c[1]:5.1f}%)")
    print("\nReal songs above each rank level (expected 10 / 5 / 1%):")
    for v, sc in scores.items():
        q = sc[2][kind == "real"]
        out[v]["levels"] = [float(np.mean(q > t)) for t in (LEVELS[0], 0.95, LEVELS[1])]
        print(f"  {v:8s} " + " / ".join(f"{100 * x:.1f}%" for x in out[v]["levels"]))
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

r"""Average over the whole song instead of 30 s: a less noisy fakeprint at test time?

Every fakeprint so far is the average spectrum of 30 s. Averaging the whole song should give a cleaner fakeprint,
especially for quiet or sparse songs. The real training songs are mostly FMA 30 s clips, so training on whole songs
would let a model learn "less noise = AI" (a shortcut). So the models stay trained on 30 s (as temporal_test.py),
and only complete test songs are scored both ways:
  newreal   the 1,077 new real songs of newreal_test.py (false alarms); their 30 s is from 40% into the song
  fresh44   Suno V5.5, MiniMax, Lyria 3 (catch rate) and the 79 Jamendo songs (false alarms); 30 s from the start
"whole" = the average spectrum of up to 8 minutes from the start of the file (at most 12 MB is read), with the same
MP3 conversion as before. Both versions are compared on exactly the same songs. Cut-offs from cross-validation on
the training songs (5% of real songs), as temporal_test.py. Models: LR per family and LR one boundary, each alone
and with the precise comb (joint).
Download: --build streams the full files (~9-10 GB, ~1.5-2 hours); only features are kept (features_tm\wholesong\).

  python -I wholesong_test.py --build --n 3 --out-dir <scratch>   # quick trial, 3 songs per source
  python -I wholesong_test.py --build
  python -I wholesong_test.py                                     # ~30 min; reports\wholesong_test.json
"""
import argparse
import io
import json
import os
import sys
import time
import urllib.parse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
MAX_SECONDS = 480.0
MAX_BYTES = 12_000_000
FRESH_AI = ["suno_v5.5", "minimax3", "lyria3", "real_jamendo"]


def jobs_for(tm, n):
    """(set, source, name, url) of every complete test song."""
    import fresh44_test as f44
    jobs = []
    for p in sorted((tm / "newreal").glob("*.npz")):
        names = np.load(p)["names"].astype(str)[: n or None]
        for nm in names:
            _, ident, fname = nm.split("/", 2)
            url = f"https://archive.org/download/{urllib.parse.quote(ident)}/{urllib.parse.quote(fname)}"
            jobs.append(("newreal", p.stem, nm, url))
    have = set(np.load(tm / "fresh44.npz")["names"].astype(str))
    for source, repo, folder, min_size, _ in f44.HF_SOURCES:
        url = f"https://huggingface.co/api/datasets/{repo}/tree/main{'/' + folder if folder else ''}?recursive=true"
        files = [x["path"] for x in json.loads(f44.http_get(url)[0]) if x["type"] == "file"
                 and x["path"].lower().endswith(".mp3") and (source != "real_jamendo" or "/audio/" in x["path"])]
        by_name = {f"{source}/{PurePosixPath(x).name}": x for x in files}
        picked = sorted(nm for nm in by_name if nm in have)[: n or None]
        base = f"https://huggingface.co/datasets/{repo}/resolve/main/"
        jobs += [("fresh44", source, nm, base + urllib.parse.quote(by_name[nm])) for nm in picked]
    return jobs


def work(job):
    """Runs in a worker: read up to MAX_BYTES of one song, whole-song fakeprints of both bands."""
    import soundfile as sf
    import detect
    import fakeprint as fpm
    import fresh44_test as f44
    which, source, name, url, bands = job
    try:
        blob, _ = f44.http_get(url, 0, MAX_BYTES - 1)
        with sf.SoundFile(io.BytesIO(blob)) as snd:
            sr = snd.samplerate
            data = snd.read(int(sr * MAX_SECONDS), dtype="float32", always_2d=True)
        if len(data) < sr * 10:
            return which, source, name, None, "shorter than 10 s", len(blob)
        out = {"src_sr": sr, "seconds": len(data) / sr}
        data = data.astype(np.float64)
        for k, b in enumerate(bands):
            freqs, spec = detect.formatted_spectrum(data, sr, b["sr"], b["mp3_level"], MAX_SECONDS)
            f, fp = fpm.fakeprint(freqs, spec, b["fmin"], b["fmax"])
            out[f"fp{k}"] = fp.astype(np.float32)
            out[f"sharp{k}"], out[f"spacing{k}"] = fpm.precise_comb(f, fp)
        return which, source, name, out, "", len(blob)
    except Exception as exc:
        return which, source, name, None, str(exc)[:120], 0


def build(out_dir, n, workers):
    import joblib
    import fresh44_test as f44
    out_dir.mkdir(parents=True, exist_ok=True)
    two = joblib.load(HERE / "model_twoband.joblib")
    bands = [{k: v for k, v in b.items() if k != "freqs"} for b in two["bands"]]
    print("Picking songs ...")
    jobs = jobs_for(HERE / "features_tm", n)
    print(f"  {len(jobs)} songs: " + ", ".join(f"{s} {sum(j[1] == s for j in jobs)}" for s in dict.fromkeys(j[1] for j in jobs)))
    res, fetched, t0 = {"newreal": [], "fresh44": []}, 0, time.time()
    with ProcessPoolExecutor(workers, initializer=f44.quiet_worker) as pool:
        futs = [pool.submit(work, j + (bands,)) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            which, source, name, out, err, got = fut.result()
            fetched += got
            if out is None:
                print(f"  skip {name}: {err}")
            else:
                res[which].append((source, name, out))
            if k % 50 == 0:
                spent = time.time() - t0
                print(f"  {k} / {len(jobs)}, {fetched / 1e9:.2f} GB, {spent / 60:.1f} min, "
                      f"about {spent / k * (len(jobs) - k) / 60:.0f} min left", flush=True)
    for which, rows in res.items():
        if not rows:
            continue
        rows.sort(key=lambda r: r[1])
        feats = {key: np.array([r[2][key] for r in rows]) for key in rows[0][2]}
        np.savez_compressed(out_dir / f"{which}.npz", names=np.array([r[1] for r in rows]),
                            sources=np.array([r[0] for r in rows]), **feats)
        print(f"Saved {len(rows)} {which} songs (median {np.median(feats['seconds']):.0f} s used)")
    print(f"{fetched / 1e9:.2f} GB downloaded in {(time.time() - t0) / 60:.1f} min; no audio kept.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--n", type=int, default=0, help="songs per source, for a quick trial")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--features", default=str(HERE / "features_tm"))
    ap.add_argument("--out-dir", default=str(HERE / "features_tm" / "wholesong"))
    ap.add_argument("--out", default=str(HERE / "reports" / "wholesong_test.json"))
    args = ap.parse_args()
    if args.build:
        build(Path(args.out_dir), args.n, args.workers)
        return

    import temporal_test as tt
    from nonlinear_test import NAMES
    tm = Path(args.features)
    print("Loading ...")
    d, _, f = tt.load(tm, 0)
    short = {}  # the 30 s features of the same songs, by name
    for p in sorted((tm / "newreal").glob("*.npz")):
        x = dict(np.load(p))
        short |= {nm: (x["fp0"][i], x["fp1"][i], x["sharp0"][i], x["sharp1"][i])
                  for i, nm in enumerate(x["names"].astype(str))}
    x = dict(np.load(tm / "fresh44.npz"))
    short |= {nm: (x["fp0"][i], x["fp1"][i], x["sharp0"][i], x["sharp1"][i])
              for i, nm in enumerate(x["names"].astype(str))}
    held = {}
    for which in ("newreal", "fresh44"):
        w = dict(np.load(Path(args.out_dir) / f"{which}.npz"))
        names, src = w["names"].astype(str), w["sources"].astype(str)
        ok = (w["src_sr"] >= 44100) & np.isin(names, list(short))
        for s in sorted(set(src[ok])):
            m = ok & (src == s)
            held[f"{s} whole"] = {"X": np.hstack([w["fp0"][m], w["fp1"][m]]), "sharp0": w["sharp0"][m],
                                  "sharp1": w["sharp1"][m], "sources": src[m]}
            sh = [short[nm] for nm in names[m]]
            held[f"{s} 30 s"] = {"X": np.array([np.concatenate([a[0], a[1]]) for a in sh]),
                                 "sharp0": np.array([a[2] for a in sh]), "sharp1": np.array([a[3] for a in sh]),
                                 "sources": src[m]}
            print(f"  {s}: {int(m.sum())} songs, median {np.median(w['seconds'][m]):.0f} s in the whole-song version")
    X = np.hstack([d["fp0"], d["fp1"]]).astype(np.float32)
    Xf = np.hstack([f["fp0"], f["fp1"]]).astype(np.float32)
    out = Path(args.out)
    results = json.loads(out.read_text()) if out.exists() else {}  # a re-run keeps the finished models
    results.pop("recalibrated", None)
    for kind in ("family", "lr"):
        if kind in results:
            continue
        print(f"\n===== {NAMES[kind]} =====", flush=True)
        r = tt.evaluate(kind, X, d, {h: v["X"] for h, v in held.items()}, held, Xf, f, ["cv", "fresh"])
        results[kind] = {h: r["fresh"][f"held {h}"] for h in held}
        Path(args.out).write_text(json.dumps(results, indent=1))

    real = [s for s in dict.fromkeys(h.rsplit(" ", 1)[0].replace(" 30", "") for h in held) if s in
            ("community", "jamendo", "mixtapes", "real_jamendo")]
    ai = [s for s in ("suno_v5.5", "minimax3", "lyria3") if f"{s} whole" in held]
    for kind, r in results.items():
        print(f"\n{NAMES[kind]}: flagged, classifier (with comb)     30 s            whole song")
        for s in real + ai:
            a, b = r[f"{s} 30 s"], r[f"{s} whole"]
            print(f"  {s:14s} {'real' if s in real else 'AI  '} ({a[2]:4d})   {100 * a[0]:5.1f}% ({100 * a[1]:4.1f}%)"
                  f"    {100 * b[0]:5.1f}% ({100 * b[1]:4.1f}%)")
        for label, group in (("ALL REAL", real), ("ALL AI", ai)):
            nn = np.array([r[f"{s} whole"][2] for s in group])
            for v in ("30 s", "whole"):
                vals = [float(nn @ [r[f"{s} {v}"][j] for s in group] / nn.sum()) for j in (0, 1)]
                results[kind][f"{label} {v}"] = vals + [int(nn.sum())]
            a, b = results[kind][f"{label} 30 s"], results[kind][f"{label} whole"]
            print(f"  {label:19s} ({int(nn.sum()):4d})   {100 * a[0]:5.1f}% ({100 * a[1]:4.1f}%)    "
                  f"{100 * b[0]:5.1f}% ({100 * b[1]:4.1f}%)")
    Path(args.out).write_text(json.dumps(results, indent=1))
    results["recalibrated"] = recalibrated(d, X, held, real, ai, NAMES)
    Path(args.out).write_text(json.dumps(results, indent=1))


def recalibrated(d, X, held, real, ai, names):
    """Whole-song scores are higher for every song (a cleaner fakeprint), so the 30 s cut-off does not fit them.
    Fair comparison: each version gets its own cut-off at 5% of real songs of its own kind. The complete real test
    songs are split in two halves (by song, seed 0): one half sets the cut-offs (and, for the joint rule, the
    real-song references), the other half measures false alarms. The AI songs are scored with those cut-offs."""
    import temporal_test as tt
    y = d["sources"] != "real"
    rng = np.random.default_rng(0)
    out = {}
    for kind in ("family", "lr"):
        score = tt.train(kind, X, y, d["fam"], np.ones(len(y), bool))
        out[kind] = {}
        print(f"\n{names[kind]}, own cut-off per version (5% of half the real test songs): flagged, classifier (joint)")
        for v in ("30 s", "whole"):
            p = {s: score(held[f"{s} {v}"]["X"]) for s in real + ai}
            r_p = np.concatenate([p[s] for s in real])
            r_16 = np.concatenate([held[f"{s} {v}"]["sharp0"] for s in real])
            r_44 = np.concatenate([held[f"{s} {v}"]["sharp1"] for s in real])
            half = rng.permutation(len(r_p)) < len(r_p) // 2  # True: sets the cut-offs
            refs = {"clf": np.sort(r_p[half]), "s16": np.sort(r_16[half]), "s44": np.sort(r_44[half])}
            jnt = lambda pp, a, b: tt.joint_score(refs, pp, a, b)  # noqa: E731
            cut, jcut = np.quantile(r_p[half], 0.95), np.quantile(jnt(r_p[half], r_16[half], r_44[half]), 0.95)
            res = {"real (other half)": [float(np.mean(r_p[~half] > cut)),
                                         float(np.mean(jnt(r_p[~half], r_16[~half], r_44[~half]) > jcut)),
                                         int((~half).sum())]}
            for s in ai:
                h = held[f"{s} {v}"]
                res[s] = [float(np.mean(p[s] > cut)), float(np.mean(jnt(p[s], h["sharp0"], h["sharp1"]) > jcut)),
                          len(p[s])]
            nn = np.array([res[s][2] for s in ai])
            res["ALL AI"] = [float(nn @ [res[s][j] for s in ai] / nn.sum()) for j in (0, 1)] + [int(nn.sum())]
            out[kind][v] = res
        for row in out[kind]["30 s"]:
            a, b = out[kind]["30 s"][row], out[kind]["whole"][row]
            print(f"  {row:18s} ({a[2]:4d})   30 s {100 * a[0]:5.1f}% ({100 * a[1]:4.1f}%)    whole {100 * b[0]:5.1f}% "
                  f"({100 * b[1]:4.1f}%)")
    return out


if __name__ == "__main__":
    main()

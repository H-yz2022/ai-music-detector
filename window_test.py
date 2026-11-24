r"""Score more than the first 30 s: does using several windows of a song help?

Whole songs (read into memory, never saved), all never trained on:
- real: the 79 Jamendo songs of fresh44, and 40 FMA tracks of fresh44 read whole from fma_full.zip;
- AI: 40 Suno V5.5 and 40 MiniMax Music 3.0 (44.1 kHz files) of fresh44.
Each song is cut into 30 s windows (0-30 s, 30-60 s, ... up to 8 windows = 4 min; a last part shorter than 30 s
is dropped), and every window gets predict's two fakeprints. The whole song (first 4 min) also gets one fakeprint
from its average spectrum. Rules compared, all with the default model (model_twoband.joblib) and its saved
cut-offs:
  first 30 s       what predict does now
  middle window    the window in the middle of the song
  mean / median    of P(AI) over all windows
  whole song       one fakeprint of the whole song
The joint rule uses the same aggregation for the combs. AUC (no cut-off needed) is shown too.

  python -I window_test.py                            # ~1.1 GB read; reports\window_test.json
  python -I window_test.py --n 2 --out <scratch dir>  # quick trial
  python -I window_test.py --rescore                  # score the saved fakeprints again
Fakeprints go to features_hq\unseen\windows.npz (float16).
"""
import argparse
import io
import json
import os
import random
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import detect  # noqa: E402
import fakeprint as fpm  # noqa: E402
from fresh44_test import HF_SOURCES, RemoteFile, hf_jobs, http_get, quiet_worker, zip_member  # noqa: E402

HERE = Path(__file__).resolve().parent
FMA_FULL = "https://os.unil.cloud.switch.ch/fma/fma_full.zip"
PICK = {"real_fma": 40, "real_jamendo": 79, "suno_v5.5": 40, "minimax3": 40}
WIN, MAX_WIN = 30.0, 8
RULES = ["first 30 s", "middle window", "mean", "median", "whole song"]


def features(x, sr, bands, seconds):
    out = {}
    for k, b in enumerate(bands):
        freqs, spec = detect.formatted_spectrum(x, sr, b["sr"], b["mp3_level"], seconds)
        f, fp = fpm.fakeprint(freqs, spec, b["fmin"], b["fmax"])
        out[f"fp{k}"] = fp.astype(np.float32)
        out[f"sharp{k}"], _ = fpm.precise_comb(f, fp)
    return out


def work(job):
    """Runs in a worker process: read one whole song, fakeprint each 30 s window and the whole song."""
    source, name, where, bands = job
    try:
        blob, got = zip_member(*where[1:]) if where[0] == "zip" else (lambda b: (b, len(b)))(http_get(where[1])[0])
        with sf.SoundFile(io.BytesIO(blob)) as snd:
            sr = snd.samplerate
            x = snd.read(int(sr * WIN * MAX_WIN), dtype="float64", always_2d=True)
        del blob
        n = min(MAX_WIN, int(len(x) // (sr * WIN)))
        if n < 2:
            return source, name, None, f"only {len(x) / sr:.0f} s", got
        x = x[:int(n * WIN * sr)]
        wins = [features(x[int(k * WIN * sr):int((k + 1) * WIN * sr)], sr, bands, WIN) for k in range(n)]
        whole = features(x, sr, bands, n * WIN)
        return source, name, {"src_sr": sr, "wins": wins, "whole": whole, "seconds": len(x) / sr}, "", got
    except Exception as exc:
        return source, name, None, f"{type(exc).__name__}: {exc}", 0


def plan(n, seed):
    fresh = np.load(HERE / "features_hq" / "unseen" / "fresh44.npz")
    rng = random.Random(seed)
    want = {}
    for s in PICK:
        names = sorted(str(m) for m, src, sr in zip(fresh["names"], fresh["sources"], fresh["src_sr"])
                       if src == s and sr >= 44100)
        want[s] = set(rng.sample(names, min(len(names), n or PICK[s])))
    ids = {PurePosixPath(m).stem for m in want["real_fma"]}  # FMA track ids: the same tracks, whole
    jobs = []
    big = RemoteFile(FMA_FULL)
    for i in zipfile.ZipFile(big).infolist():
        stem = PurePosixPath(i.filename).stem
        if i.filename.endswith(".mp3") and stem in ids:
            jobs.append(("real_fma", f"fma_full/{stem}.mp3", ("zip", FMA_FULL, i.header_offset, i.compress_size,
                                                             i.compress_type, len(i.filename.encode()))))
    fetched = big.fetched
    for source, repo, folder, min_size, _ in HF_SOURCES:
        if source in want:
            jobs += [(s, nm, ("http", w[1])) for s, nm, w in hf_jobs(source, repo, folder, min_size, 0, seed)
                     if nm in want[source]]
    print("Songs: " + ", ".join(f"{s} {sum(j[0] == s for j in jobs)}" for s in PICK))
    return jobs, fetched


def collect(args, bands):
    jobs, fetched = plan(args.n, args.seed)
    bands = [{k: v for k, v in b.items() if k != "freqs"} for b in bands]
    print(f"Reading {len(jobs)} whole songs with {args.workers} workers ...")
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
                print(f"  {k} / {len(jobs)}, {fetched / 1e6:.0f} MB read, {spent / 60:.1f} min", flush=True)
    res.sort(key=lambda r: r[1])
    nw = np.array([len(r[2]["wins"]) for r in res])
    d = {"names": np.array([r[1] for r in res]), "sources": np.array([r[0] for r in res]), "n_windows": nw,
         "src_sr": np.array([r[2]["src_sr"] for r in res]), "read_mb": fetched / 1e6}
    for key in ("fp0", "fp1", "sharp0", "sharp1"):
        first = res[0][2]["whole"][key]
        arr = np.full((len(res), MAX_WIN) + np.shape(first), np.nan, dtype=np.float32)
        for i, r in enumerate(res):
            for k, w in enumerate(r[2]["wins"]):
                arr[i, k] = w[key]
        d[f"win_{key}"] = arr
        d[f"whole_{key}"] = np.array([r[2]["whole"][key] for r in res], dtype=np.float32)
    Path(args.features).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.features, **{k: v.astype(np.float16) if "_fp" in k else v for k, v in d.items()})
    print(f"Done: {fetched / 1e6:.0f} MB read in {(time.time() - t0) / 60:.1f} min, no audio kept. "
          f"Fakeprints -> {args.features}")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--model", default=str(HERE / "model_twoband.joblib"))
    args = ap.parse_args()
    out = Path(args.out) if args.out else None
    args.features = str((out or HERE / "features_hq" / "unseen") / "windows.npz")
    report = (out or HERE / "reports") / "window_test.json"
    two = joblib.load(args.model)
    d = {k: (v.astype(np.float32) if "_fp" in k else v) for k, v in np.load(args.features).items()} \
        if args.rescore else collect(args, two["bands"])

    src, nw = d["sources"].astype(str), d["n_windows"]
    n, m = len(src), d["win_fp0"].shape[1]
    ok = np.arange(m)[None, :] < nw[:, None]
    X = np.concatenate([d["win_fp0"], d["win_fp1"]], axis=2)
    p = np.full((n, m), np.nan)
    p[ok] = detect.ai_proba(two["model"], X[ok])
    s16, s44 = d["win_sharp0"], d["win_sharp1"]
    mid = nw // 2
    r = np.arange(n)
    pw = detect.ai_proba(two["model"], np.hstack([d["whole_fp0"], d["whole_fp1"]]))
    agg = {"first 30 s": (p[:, 0], s16[:, 0], s44[:, 0]),
           "middle window": (p[r, mid], s16[r, mid], s44[r, mid]),
           "mean": (np.nanmean(p, 1), np.nanmean(s16, 1), np.nanmean(s44, 1)),
           "median": (np.nanmedian(p, 1), np.nanmedian(s16, 1), np.nanmedian(s44, 1)),
           "whole song": (pw, d["whole_sharp0"], d["whole_sharp1"])}
    real = np.char.startswith(src, "real")
    groups = {"real_fma": src == "real_fma", "real_jamendo": src == "real_jamendo", "all real": real,
              "suno_v5.5": src == "suno_v5.5", "minimax3": src == "minimax3"}
    results = {}
    print(f"\n{n} songs: " + ", ".join(f"{g} {int(k.sum())}" for g, k in groups.items())
          + f"; windows per song: median {np.median(nw):.0f}" + (f"; {float(d['read_mb']):.0f} MB read" if "read_mb" in d else ""))
    print(f"Cut-offs (saved): P(AI) > {two['cutoff']:.3f}, joint {two['joint_cutoff']:.4f}")
    print(f"\n{'rule':15s}{'FMA':>7s}{'Jamendo':>9s}{'real':>7s}{'Suno':>7s}{'MiniMax':>9s}{'AUC':>7s}   "
          f"joint: {'real':>5s}{'Suno':>7s}{'MiniMax':>9s}")
    for rule, (pr, a, b) in agg.items():
        flag = pr > two["cutoff"]
        jflag = detect.joint_score(two["real_scores"], pr, a, b) > two["joint_cutoff"]
        res = {g: round(100 * float(flag[k].mean()), 1) for g, k in groups.items()}
        jres = {g: round(100 * float(jflag[k].mean()), 1) for g, k in groups.items()}
        auc = roc_auc_score(~real, pr)
        results[rule] = {"classifier": res, "joint": jres, "auc": round(float(auc), 4)}
        print(f"{rule:15s}{res['real_fma']:7.1f}{res['real_jamendo']:9.1f}{res['all real']:7.1f}{res['suno_v5.5']:7.1f}"
              f"{res['minimax3']:9.1f}{auc:7.3f}   joint: {jres['all real']:5.1f}{jres['suno_v5.5']:7.1f}"
              f"{jres['minimax3']:9.1f}")
    # how much does P(AI) change from window to window?
    rate = lambda sel, k: round(100 * float(np.mean(p[sel & (nw > k), k] > two["cutoff"])), 1) if (sel & (nw > k)).any() else None  # noqa: E731
    pos = [rate(real, k) for k in range(m)]
    print("\nReal songs flagged per window position (0-30 s, 30-60 s, ...): " + ", ".join(map(str, pos)))
    pos_ai = [rate(~real, k) for k in range(m)]
    print("AI songs caught per window position: " + ", ".join(map(str, pos_ai)))
    anyw = np.nanmax(np.where(ok, p, -1), 1) > two["cutoff"]
    print(f"Real songs with at least one window flagged: {100 * anyw[real].mean():.1f}%")
    results["per_window_real"], results["per_window_ai"] = pos, pos_ai
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"songs": {g: int(k.sum()) for g, k in groups.items()}, "results": results}, indent=1))
    print(f"Saved {report}")


if __name__ == "__main__":
    main()

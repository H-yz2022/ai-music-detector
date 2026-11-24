r"""Does the fakeprint throw away useful information? Variants of the 5-16 kHz feature, from the stored spectra.

The fakeprint (fakeprint.fakeprint) takes the average dB spectrum, subtracts its lower envelope, clips to 0..5 dB
and divides by its maximum. Each step drops something. Variants, all from features_hq\ (FMA small + Echoes, the
12,357 train_twoband.py songs; no download):
  fakeprint      as now
  no scaling     the same in dB, not divided by its maximum (keeps how strong the peaks are)
  no clip        not divided, and not clipped at 5 dB (keeps very strong peaks)
  peaks + dips   spectrum minus its 31-bin running median, not clipped at 0 either (keeps dips as well as peaks)
  + shape        fakeprint plus the coarse spectrum 0-22 kHz in 500 Hz bands, relative to its 5-8 kHz level
                 (keeps the overall shape, e.g. where a generator cuts off)
Each variant goes through the same logistic regression (balanced classes), at 5% false alarms on real songs:
  cv     5-fold cross-validation, real songs grouped by artist
  logo   leave one generator out (12 Echoes generators); half the held-out real fold sets the cut-off, the other
         half measures false alarms
Only the 5-16 kHz band, so the numbers are below the two-band model's; what matters is the difference between
variants. Caveat: these real songs are all FMA small; a variant that learns their look (level, shape) would also
need a test on other real songs.

  python -I fpinfo_test.py      # reports\fpinfo_test.json
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold  # noqa: E402

import fakeprint as fpm  # noqa: E402
from cutoff_test import artists_of  # noqa: E402
from logo_test import fit  # noqa: E402

HERE = Path(__file__).resolve().parent


def variants(freqs, spec, coarse, edges):
    band = (freqs > 5000) & (freqs < 16000)
    out = {k: [] for k in ("fakeprint", "no scaling", "no clip", "peaks + dips")}
    for s in spec.astype(np.float64):
        c = s[band]
        hull = np.clip(fpm.lower_hull(c, 10), -45.0, None)
        raw = np.clip(c - hull, 0.0, None)
        cl = np.minimum(raw, 5.0)
        out["fakeprint"].append(cl / (1e-6 + cl.max()))
        out["no scaling"].append(cl / 5.0)
        out["no clip"].append(raw / 5.0)
        out["peaks + dips"].append((c - median_filter(c, 31, mode="nearest")) / 5.0)
    out = {k: np.array(v, np.float32) for k, v in out.items()}
    lo = edges[:-1]
    rel = coarse - coarse[:, (lo >= 5000) & (lo < 8000)].mean(1, keepdims=True)
    rel = (rel - rel.mean(0)) / (rel.std(0) + 1e-6)
    out["+ shape"] = np.hstack([out["fakeprint"], rel.astype(np.float32)])
    return out


def main():
    print("Loading ...")
    parts = [np.load(HERE / "features_hq" / f) for f in ("fma_small.npz", "echoes.npz")]
    lq = set(np.load(HERE / "features" / "fma_small.npz")["names"].astype(str)) | \
        set(np.load(HERE / "features" / "unseen" / "echoes.npz")["names"].astype(str))
    names = np.concatenate([p["names"] for p in parts]).astype(str)
    keep = np.isin(names, list(lq))  # the train_twoband.py songs
    spec = np.concatenate([p["spec"] for p in parts])[keep]
    coarse = np.concatenate([p["coarse"] for p in parts])[keep]
    labels = np.concatenate([p["gens"] for p in parts]).astype(str)[keep]
    names = names[keep]
    real, y = labels == "real", labels != "real"
    art = artists_of(names, labels, str(HERE / "features" / "fma_artists.csv"))
    groups = np.array([f"artist{a}" if a is not None else n for a, n in zip(art, names)], dtype=object)
    feats = variants(parts[0]["freqs"], spec, coarse, parts[0]["coarse_edges"])
    del spec
    gens = sorted(set(labels[y]))
    print(f"  {len(labels)} songs: {int(real.sum())} real, {int(y.sum())} AI from {len(gens)} generators")

    real_idx = np.flatnonzero(real)
    fold = np.empty(len(real_idx), int)
    for k, (_, te) in enumerate(GroupKFold(len(gens)).split(real_idx, groups=groups[real])):
        fold[te] = k
    results = {}
    for name, X in feats.items():
        t = time.time()
        oof = np.zeros(len(y))
        for _, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, labels, groups):
            use = np.ones(len(y), bool)
            use[te] = False
            oof[te] = fit(X, y, use).predict_proba(X[te])[:, 1]
        cut = np.quantile(oof[real], 0.95)
        r = {"cv": {g: float(np.mean(oof[labels == g] > cut)) for g in gens}}
        r["cv"]["ALL AI"] = float(np.mean(oof[y] > cut))
        rng = np.random.default_rng(0)
        r["logo"], fa, n = {}, [], []
        for i, g in enumerate(gens):
            cal = rng.permutation(real_idx[fold == i])
            set_cut, check = cal[: len(cal) // 2], cal[len(cal) // 2:]
            use = labels != g
            use[cal] = False
            m = fit(X, y, use)
            c = np.quantile(m.predict_proba(X[set_cut])[:, 1], 0.95)
            r["logo"][g] = float(np.mean(m.predict_proba(X[labels == g])[:, 1] > c))
            fa.append(float(np.mean(m.predict_proba(X[check])[:, 1] > c)))
            n.append(int((labels == g).sum()))
        r["logo"]["ALL AI"] = float(np.dot(n, [r["logo"][g] for g in gens]) / sum(n))
        r["logo"]["REAL flagged"] = float(np.mean(fa))
        results[name] = r
        print(f"  {name:13s} cv {100 * r['cv']['ALL AI']:5.1f}%   left out {100 * r['logo']['ALL AI']:5.1f}% "
              f"(real flagged {100 * r['logo']['REAL flagged']:.1f}%)   {(time.time() - t) / 60:.1f} min", flush=True)

    print(f"\n{'generator':14s}" + "".join(f"{k:>15s}" for k in feats) + "   (cv / left out)")
    for g in gens + ["ALL AI"]:
        print(f"{g:14s}" + "".join(f"{100 * results[k]['cv'][g]:7.0f} /{100 * results[k]['logo'][g]:4.0f}%"
                                   for k in feats))
    out = HERE / "reports" / "fpinfo_test.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

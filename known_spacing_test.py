r"""Step 2: a comb check aimed at one known spacing (e.g. Mureka's 300 Hz) instead of searching for any comb.

fullband_probe.py found that Mureka's songs share a comb at 300 Hz (5-16 kHz), too faint to see in one song with
the precise comb, which searches every spacing from 20 to 2000 Hz and keeps the best (so real songs also get a high
"best" by chance). If the spacing is known, only that one spacing is checked: no search, so less chance for real
songs to look like a comb.
Known-spacing score of a song at spacing s0: the bump-removed 5-16 kHz fakeprint (as precise_comb) averaged at
all multiples of each candidate spacing within +-0.3% of s0 (200 candidates, the "harmonic sum"); the score is
the z-score of the candidates within 1/4 bin (at the top multiple) of s0, against all 200.
No download: stored 30 s fakeprints (features_tm\, as temporal_test.py), 44.1 kHz files or more.
Per AI family: its songs are split in half by group. The spacing comes from the training half (the precise comb
of its mean fakeprint); the test half is scored. A family enters the "bank" only if that mean fakeprint has a
clear comb (precise comb >= 8; real songs' mean fakeprint: about 4.4).
Compared at 5% of real training songs flagged (cut-offs from them):
  precise   the precise comb now in predict (best of any spacing)
  known     the known-spacing score for the family's own spacing (one family at a time; an upper bound)
  bank      the highest known-spacing score over all bank spacings (each as a share of real songs below it)
  clf       LR one boundary (predict's model), 5-fold cross-validated (real songs grouped by artist)
  clf+precise, clf+bank: the highest of the two shares of real songs below (as the joint rule)
False alarms are also checked on real songs never used for any cut-off: fresh44's, the 1,077 new ones, live
concerts.
Rules fixed before running: "known" helps Mureka if it catches at least 15 points more of Mureka's test half than
"precise". clf+bank is better than clf+precise if it catches at least 2 points more of all AI test halves and flags
at most 6% of the never-used real songs.

  python -I known_spacing_test.py        # ~20 min; reports\known_spacing_test.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold  # noqa: E402

import fakeprint as fpm  # noqa: E402
import temporal_test as tt  # noqa: E402
from nonlinear_test import pct  # noqa: E402

tt.SKIP |= {"tstd0", "tstd1", "tco0", "tco1", "shape0", "shape1", "stereo"}
HERE = Path(__file__).resolve().parent
N_CAND, WIDTH = 200, 0.003  # candidates, +-0.3% around the known spacing
MIN_BANK = 8.0


def bump_removed(fp, f):
    k = max(3, int(round(fpm.BUMP_HZ / (f[1] - f[0]))) | 1)
    return np.clip(fp - uniform_filter1d(fp, k, axis=1, mode="nearest"), 0.0, None).astype(np.float32)


def known_score(x, f, s0):
    """Known-spacing z-score of each row of x (bump-removed fakeprints on bins f) at spacing s0."""
    df = f[1] - f[0]
    cands = s0 * (1 + np.linspace(-WIDTH, WIDTH, N_CAND))
    k = np.arange(1, int(f[-1] // (s0 * (1 - WIDTH))) + 2)
    chunk = max(10, int(1e7 // (N_CAND * len(k))))  # about 40 MB per temporary array
    pos = (cands[:, None] * k[None, :] - f[0]) / df  # in bins
    inside = (pos >= 0) & (pos <= len(f) - 1)
    i0 = np.clip(np.floor(pos).astype(np.int64), 0, len(f) - 2)
    w = np.clip(pos - i0, 0, 1).astype(np.float32)
    cnt = np.maximum(inside.sum(1), 1)
    near = np.abs(cands - s0) <= 0.25 * df / k[-1] + 1e-9
    near[np.argmin(np.abs(cands - s0))] = True
    out = np.empty(len(x), np.float32)
    for a in range(0, len(x), chunk):
        xb = x[a:a + chunk]
        v = (1 - w) * xb[:, i0] + w * xb[:, i0 + 1]  # rows x cands x multiples
        h = np.where(inside, v, 0).sum(2) / cnt
        z = (h - h.mean(1, keepdims=True)) / (h.std(1, keepdims=True) + 1e-12)
        out[a:a + chunk] = z[:, near].max(1)
    return out


def main():
    print("Loading ...", flush=True)
    d, held, fr = tt.load(HERE / "features_tm", 0)
    for p in sorted((HERE / "features_tm" / "newreal").glob("*.npz")):
        x = np.load(p)
        ok = x["src_sr"] >= 44100
        held[p.stem] = {k: x[k][ok] for k in ("fp0", "fp1", "sharp0", "sharp1")}
    f = np.asarray(joblib.load(HERE / "model_twoband.joblib")["bands"][1]["freqs"], dtype=np.float64)
    labels, groups, fam = d["sources"], d["groups"].astype(str), d["fam"]
    real, y = labels == "real", labels != "real"
    print("Bump removal ...", flush=True)
    xb = bump_removed(d["fp1"], f)
    hx = {h: bump_removed(v["fp1"], f) for h, v in held.items()}
    fresh_real = np.char.startswith(fr["sources"].astype(str), "real")
    hx["fresh44 real"] = bump_removed(fr["fp1"][fresh_real], f)
    held_sharp = {h: v["sharp1"] for h, v in held.items()} | {"fresh44 real": fr["sharp1"][fresh_real]}

    # Per family: spacing from the training half's mean fakeprint, test half scored
    test = np.zeros(len(y), bool)
    bank, res = {}, {"families": {}}
    for g in sorted(set(fam[y])):
        gi = np.flatnonzero(fam == g)
        tr, te = next(GroupShuffleSplit(1, test_size=0.5, random_state=0).split(gi, groups=groups[gi]))
        test[gi[te]] = True
        m_sharp, s0 = fpm.precise_comb(f, d["fp1"][gi[tr]].mean(0))
        res["families"][g] = {"spacing": float(s0), "mean_fp_comb": float(m_sharp), "test_songs": int(len(te))}
        if m_sharp >= MIN_BANK:
            bank[g] = float(s0)
        print(f"  {g:12s} mean-fp comb {m_sharp:5.1f} at {s0:8.3f} Hz{'  -> bank' if g in bank else ''}", flush=True)
    res["bank"] = bank

    print("Known-spacing scores ...", flush=True)
    ks = {g: known_score(xb, f, s0) for g, s0 in bank.items()}
    ks_held = {h: {g: known_score(v, f, s0) for g, s0 in bank.items()} for h, v in hx.items()}
    real_ks = {g: np.sort(v[real]) for g, v in ks.items()}

    def bank_share(scores):  # highest share of real training songs below, over the bank spacings
        return np.max([pct(real_ks[g], scores[g]) for g in bank], 0)
    b_all = bank_share(ks)
    b_held = {h: bank_share(v) for h, v in ks_held.items()}

    print("Classifier (5-fold cv) ...", flush=True)
    X = np.hstack([d["fp0"], d["fp1"]]).astype(np.float32)
    oof = np.zeros(len(y))
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, labels, groups), 1):
        use = np.ones(len(y), bool)
        use[te] = False
        oof[te] = tt.train("lr", X, y, fam, use)(X[te])
        print(f"    fold {k}", flush=True)
    full = tt.train("lr", X, y, fam, np.ones(len(y), bool))
    clf_held = {h: full(np.hstack([held[h]["fp0"], held[h]["fp1"]]).astype(np.float32)) for h in held}
    clf_held["fresh44 real"] = full(np.hstack([fr["fp0"][fresh_real], fr["fp1"][fresh_real]]).astype(np.float32))

    s44 = d["sharp1"]
    ref = {"clf": np.sort(oof[real]), "precise": np.sort(s44[real]), "bank": np.sort(b_all[real])}
    scores = {"precise": pct(ref["precise"], s44), "bank": pct(ref["bank"], b_all), "clf": pct(ref["clf"], oof)}
    scores["clf+precise"] = np.maximum(scores["clf"], scores["precise"])
    scores["clf+bank"] = np.maximum(scores["clf"], scores["bank"])
    cut = {m: float(np.quantile(v[real], 0.95)) for m, v in scores.items()}
    hs = {}
    for h in hx:
        p = {"precise": pct(ref["precise"], held_sharp[h]), "bank": pct(ref["bank"], b_held[h]),
             "clf": pct(ref["clf"], clf_held[h])}
        p["clf+precise"] = np.maximum(p["clf"], p["precise"])
        p["clf+bank"] = np.maximum(p["clf"], p["bank"])
        hs[h] = p
    methods = ["precise", "bank", "clf", "clf+precise", "clf+bank"]
    rows = {}
    for g in sorted(set(fam[y])):
        m = test & (fam == g)
        rows[g] = {meth: float(np.mean(scores[meth][m] > cut[meth])) for meth in methods}
        if g in bank:
            kcut = np.quantile(ks[g][real], 0.95)
            rows[g]["known"] = float(np.mean(ks[g][m] > kcut))
    rows["ALL AI (test halves)"] = {meth: float(np.mean(scores[meth][test] > cut[meth])) for meth in methods}
    never = ["fresh44 real", "jamendo", "mixtapes", "community", "etree"]
    for h in never:
        rows[f"real {h}"] = {meth: float(np.mean(hs[h][meth] > cut[meth])) for meth in methods}
    n = np.array([len(hx[h]) for h in never])
    rows["REAL never used"] = {meth: float(n @ [rows[f"real {h}"][meth] for h in never] / n.sum()) for meth in methods}
    res["rows"] = rows
    (HERE / "reports" / "known_spacing_test.json").write_text(json.dumps(res, indent=1))

    print(f"\n{'':26s}" + "".join(f"{m:>13s}" for m in ["known"] + methods))
    for r, v in rows.items():
        print(f"{r:26s}" + "".join(f"{100 * v[m]:12.1f}%" if m in v else f"{'-':>13s}" for m in ["known"] + methods))
    mu = rows.get("mureka", {})
    if "known" in mu:
        print(f"\nRule 1 (Mureka): known {100 * mu['known']:.1f}% vs precise {100 * mu['precise']:.1f}% -> "
              f"{'HELPS' if mu['known'] - mu['precise'] >= 0.15 else 'DOES NOT HELP'}")
    else:
        print("\nRule 1 (Mureka): no clear comb in the training half's mean fakeprint -> not in the bank")
    a, r = rows["ALL AI (test halves)"], rows["REAL never used"]
    print(f"Rule 2: clf+bank {100 * a['clf+bank']:.1f}% vs clf+precise {100 * a['clf+precise']:.1f}% of AI; never-used real "
          f"flagged {100 * r['clf+bank']:.1f}% -> "
          f"{'BETTER' if a['clf+bank'] - a['clf+precise'] >= 0.02 and r['clf+bank'] <= 0.06 else 'NOT BETTER'}")


if __name__ == "__main__":
    main()

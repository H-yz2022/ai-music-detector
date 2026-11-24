r"""Can training on many generators, or shift-proof features, catch a generator never seen?

Leave-one-generator-out (LOGO) on the 12 Echoes generators: for each generator g,
the classifier is trained on real FMA clips + SONICS + the other 11 Echoes
generators, then tested on g. The real FMA clips are split into 12 artist groups;
the group held out with g sets the cut-off (5% of those real songs flagged), so
the cut-off never uses training songs or g. The 100 fresh real songs
(data_fresh) give an independent false-alarm rate.

Feature sets (fixed before the run):
- fakeprint: the paper's features (where the peaks are);
- autocorr: autocorrelation of the bump-removed fakeprint at 20-1000 Hz
  (how far apart the peaks are, wherever they sit: shift-proof);
- both.
"SONICS only" is the same classifier trained without any Echoes songs (the
earlier setup), for comparison. "+ precise comb" = classifier or precise comb
(fakeprint.precise_comb > detect.SHARPNESS_THRESHOLD, training-free).

  python -I logo_test.py
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

import fakeprint as fpm  # noqa: E402
from cutoff_test import artists_of  # noqa: E402
from detect import SHARPNESS_THRESHOLD  # noqa: E402
from scale_test import load_features, load_test  # noqa: E402

HERE = Path(__file__).resolve().parent


def autocorr_feats(f, fps, lo_hz=20.0, hi_hz=1000.0):
    """Autocorrelation of each bump-removed fakeprint at lags lo_hz..hi_hz (shift-proof features)."""
    df = f[1] - f[0]
    lo, hi = int(np.ceil(lo_hz / df)), int(hi_hz / df)
    out = np.empty((len(fps), hi - lo + 1), np.float32)
    for i, fp in enumerate(fps):
        out[i] = fpm._autocorr_peaks(f, fp.astype(np.float64), lo_hz, hi_hz, fpm.BUMP_HZ)[0][lo:hi + 1]
    return out


def _precise_part(job):  # worker process: loads its own slice
    path, start, stop = job
    d = np.load(path)
    return [fpm.precise_comb(d["freqs"], fp)[0] for fp in d["fp"][start:stop].astype(np.float64)]


def precise_file(path, workers, chunk=500):
    n = len(np.load(path)["gens"])
    with ProcessPoolExecutor(workers) as pool:
        parts = pool.map(_precise_part, [(str(path), i, min(i + chunk, n)) for i in range(0, n, chunk)])
        return np.array([s for part in parts for s in part])


def fit(X, y, use):
    """Logistic regression on rows where `use` is True, with balanced class weights (like detect.make_model).
    Rows not used get weight 0, which avoids copying the big feature matrix."""
    w = np.zeros(len(y))
    for cls in (False, True):
        sel = use & (y == cls)
        w[sel] = use.sum() / (2 * sel.sum())
    return LogisticRegression(max_iter=5000).fit(X, y, sample_weight=w)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=str(HERE / "features"))
    ap.add_argument("--unseen", default=str(HERE / "features" / "unseen" / "echoes.npz"))
    ap.add_argument("--test", default=str(HERE / "data_fresh"))
    ap.add_argument("--artists", default=str(HERE / "features" / "fma_artists.csv"))
    ap.add_argument("--sets", nargs="+", default=["autocorr", "fakeprint", "both"])
    ap.add_argument("--only", nargs="*", help="hold out only these generators (for a quick check)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--fmin", type=float, default=1000.0)
    ap.add_argument("--fmax", type=float, default=8000.0)
    ap.add_argument("--max-duration", type=float, default=30.0)
    args = ap.parse_args()

    print("Training pool:")
    freqs, Xp, gens, names = load_features(args.features, args)
    u = np.load(args.unseen)
    Xu, ugens = u["fp"].astype(np.float32), u["gens"]
    _, Xt, tgens, _ = load_test(Path(args.test), args)
    Xr = Xt[tgens == "real"]
    gen_list = sorted(set(ugens))
    held = args.only or gen_list

    # One matrix for everything: pool rows, then Echoes rows. Fresh real songs are scored separately.
    is_pool = np.r_[np.ones(len(gens), bool), np.zeros(len(ugens), bool)]
    lab = np.r_[gens, ugens]
    y = lab != "real"
    real_idx = np.flatnonzero(lab == "real")
    echo_idx = np.flatnonzero(~is_pool)
    art = artists_of(names, gens, args.artists)[gens == "real"]
    grp = np.array([a if a is not None else f"track{i}" for i, a in enumerate(art)], dtype=object)
    real_fold = np.empty(len(real_idx), int)
    for k, (_, te) in enumerate(GroupKFold(len(gen_list)).split(real_idx, groups=grp)):
        real_fold[te] = k

    print("Precise comb on Echoes and fresh real songs ...")
    cache = Path(args.unseen).with_name(Path(args.unseen).stem + "_precise.npy")
    sharp_u = np.load(cache) if cache.exists() else None
    if sharp_u is None or len(sharp_u) != len(ugens):
        sharp_u = precise_file(args.unseen, args.workers)
        np.save(cache, sharp_u)
    sharp_r = np.array([fpm.precise_comb(freqs, fp.astype(np.float64))[0] for fp in Xr])
    comb_u = sharp_u > SHARPNESS_THRESHOLD

    print("Autocorrelation features ...")
    A = autocorr_feats(freqs, np.concatenate([Xp, Xu]))
    Ar = autocorr_feats(freqs, Xr)
    F = np.concatenate([Xp, Xu])
    del Xp, Xu

    results = {}
    for name in args.sets:
        if name == "autocorr":
            M, Mr = A, Ar
        elif name == "fakeprint":
            M, Mr = F, Xr
        else:
            M, Mr = np.hstack([F, A]), np.hstack([Xr, Ar])
        print(f"\n[{name}] {M.shape[1]} features")
        rows = {}
        # SONICS only: no Echoes songs in training; real group 0 sets the cut-off.
        cal = real_idx[real_fold == 0]
        use = is_pool.copy()
        use[cal] = False
        m = fit(M, y, use)
        cut = np.quantile(m.predict_proba(M[cal])[:, 1], 0.95)
        pu = m.predict_proba(M[echo_idx])[:, 1] > cut
        base_fa = np.mean(m.predict_proba(Mr)[:, 1] > cut)
        for gi, g in enumerate(gen_list):
            if g not in held:
                continue
            sel = ugens == g
            cal = real_idx[real_fold == gi]
            use = lab != g
            use[cal] = False
            m = fit(M, y, use)
            cut = np.quantile(m.predict_proba(M[cal])[:, 1], 0.95)
            p = m.predict_proba(M[echo_idx[sel]])[:, 1] > cut
            fa = np.mean(m.predict_proba(Mr)[:, 1] > cut)
            fa_either = np.mean((m.predict_proba(Mr)[:, 1] > cut) | (sharp_r > SHARPNESS_THRESHOLD))
            rows[g] = (sel.sum(), pu[sel].mean(), p.mean(), (p | comb_u[sel]).mean(), fa, fa_either)
            print(f"  {g:12s} SONICS only {100 * pu[sel].mean():4.0f}%   LOGO {100 * p.mean():4.0f}%   "
                  f"LOGO + precise comb {100 * (p | comb_u[sel]).mean():4.0f}%   (fresh real flagged {100 * fa:.0f}%)")
        results[name] = (rows, base_fa)
        if name == "both":
            del M

    for name, (rows, base_fa) in results.items():
        n = np.array([r[0] for r in rows.values()])
        avg = lambda i: 100 * np.sum(n * np.array([r[i] for r in rows.values()])) / n.sum()  # noqa: E731
        print(f"\n[{name}] held-out generators: {len(rows)}, songs: {n.sum()}")
        print(f"{'generator':14s}{'songs':>6s}{'SONICS only':>13s}{'LOGO':>7s}{'LOGO + precise comb':>21s}")
        for g, r in rows.items():
            print(f"{g:14s}{r[0]:6d}{100 * r[1]:12.0f}%{100 * r[2]:6.0f}%{100 * r[3]:20.0f}%")
        print(f"{'ALL':14s}{n.sum():6d}{avg(1):12.0f}%{avg(2):6.0f}%{avg(3):20.0f}%")
        fa, fa_either = (100 * np.mean([r[i] for r in rows.values()]) for i in (4, 5))
        print(f"fresh real songs flagged: SONICS only {100 * base_fa:.0f}%, LOGO {fa:.1f}% (mean over folds), "
              f"LOGO + precise comb {fa_either:.1f}%")
    print(f"\nPrecise comb alone (no training): {100 * comb_u.mean():.0f}% of all Echoes songs, "
          f"fresh real flagged {100 * np.mean(sharp_r > SHARPNESS_THRESHOLD):.0f}%")


if __name__ == "__main__":
    main()

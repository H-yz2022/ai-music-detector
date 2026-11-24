r"""Both bands together: 1-8 kHz (16 kHz audio) + 5-16 kHz (44.1 kHz audio), on the same songs.

Same 12,357 songs as hq_test.py (7,894 real FMA clips, 4,463 Echoes songs, no SONICS).
- Classifier on both fakeprints side by side (7,167 + 4,087 features).
- Precise comb in both bands: a song counts if either band's comb stands out.
- Full detector: classifier or precise comb (either band).
Combining scores: each score is turned into "higher than what share of real songs"
(its percentile among real songs), and a song's combined score is the highest of
them. One cut-off on that flags 5% of real songs, so adding more checks does not
add false alarms. In leave-one-out, the held-out real artist group is split in
two: one half sets the cut-off, the other half measures false alarms.
Real clips are grouped by artist. Settings fixed before running.

  python -I twoband_test.py
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold  # noqa: E402

from cutoff_test import artists_of  # noqa: E402
from hq_test import load, precise_all  # noqa: E402
from logo_test import fit  # noqa: E402

HERE = Path(__file__).resolve().parent


def pct(ref, x):
    """Share of the reference (real-song) scores that each x is above: 0.97 = above 97% of real songs."""
    return np.searchsorted(np.sort(ref), x, side="right") / len(ref)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hq", nargs="+", default=[str(HERE / "features_hq" / "fma_small.npz"),
                                                str(HERE / "features_hq" / "echoes.npz")])
    ap.add_argument("--lq", nargs="+", default=[str(HERE / "features" / "fma_small.npz"),
                                                str(HERE / "features" / "unseen" / "echoes.npz")])
    ap.add_argument("--artists", default=str(HERE / "features" / "fma_artists.csv"))
    ap.add_argument("--cache", default=str(HERE / "features_hq" / "precise_cache.npz"))
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    print("Loading both bands ...")
    fh, Xh, gh, nh, _, _ = load(args.hq, "spec")
    fl, Xl, gl, nl = load(args.lq, "fp")
    common = sorted(set(nh) & set(nl))
    ih, il = {n: i for i, n in enumerate(nh)}, {n: i for i, n in enumerate(nl)}
    Xh, Xl = Xh[[ih[n] for n in common]], Xl[[il[n] for n in common]]
    labels, names = gh[[ih[n] for n in common]], np.array(common)
    real = labels == "real"
    y = ~real
    gen_list = sorted(set(labels[y]))
    print(f"  {len(names)} songs: {int(real.sum())} real, {int(y.sum())} AI")

    cache = Path(args.cache)
    c = np.load(cache) if cache.exists() else None
    if c is not None and np.array_equal(c["names"], names):
        s16, s44 = c["s16"], c["s44"]
    else:
        print("Precise comb in both bands (cached afterwards) ...")
        s16, s44 = precise_all(fl, Xl, args.workers), precise_all(fh, Xh, args.workers)
        np.savez(cache, names=names, s16=s16, s44=s44)
    X = np.hstack([Xl, Xh])
    del Xl, Xh
    print(f"  classifier features: {X.shape[1]}")

    art = artists_of(names, labels, args.artists)
    groups = np.array([f"artist{a}" if a is not None else n for a, n in zip(art, names)], dtype=object)

    # Cross-validated (every generator in training)
    print("\nCross-validation ...")
    oof = np.zeros(len(y))
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, labels, groups), 1):
        use = np.ones(len(y), bool)
        use[te] = False
        oof[te] = fit(X, y, use).predict_proba(X[te])[:, 1]
        print(f"  fold {k}")
    q = lambda v: np.quantile(v[real], 0.95)  # noqa: E731
    comb2 = np.maximum(pct(s16[real], s16), pct(s44[real], s44))
    full = np.maximum(pct(oof[real], oof), comb2)
    cv = {"clf": oof > q(oof), "comb16": s16 > q(s16), "comb44": s44 > q(s44), "comb2": comb2 > q(comb2),
          "full": full > q(full)}

    # Leave one generator out
    real_idx = np.flatnonzero(real)
    fold = np.empty(len(real_idx), int)
    for k, (_, te) in enumerate(GroupKFold(len(gen_list)).split(real_idx, groups=groups[real])):
        fold[te] = k
    rng = np.random.default_rng(0)
    logo, fa = {}, []
    print("Leave one generator out ...")
    for gi, g in enumerate(gen_list):
        cal = rng.permutation(real_idx[fold == gi])
        set_cut, check = cal[: len(cal) // 2], cal[len(cal) // 2:]
        use = labels != g
        use[cal] = False
        m = fit(X, y, use)
        ps, pg, pc = (m.predict_proba(X[i])[:, 1] for i in (set_cut, labels == g, check))
        score = lambda p, i: np.maximum.reduce([pct(ps, p), pct(s16[set_cut], s16[i]), pct(s44[set_cut], s44[i])])  # noqa: E731
        g_idx = np.flatnonzero(labels == g)
        cut_clf, cut_full = np.quantile(ps, 0.95), np.quantile(score(ps, set_cut), 0.95)
        logo[g] = (np.mean(pg > cut_clf), np.mean(score(pg, g_idx) > cut_full))
        fa.append((np.mean(pc > cut_clf), np.mean(score(pc, check) > cut_full)))
        print(f"  {g:12s} classifier {100 * logo[g][0]:4.0f}%   full detector {100 * logo[g][1]:4.0f}%")

    n = {g: int((labels == g).sum()) for g in gen_list}
    cols = ["CV clf", "LOGO clf", "comb 16k", "comb 44k", "comb both", "full CV", "full LOGO"]
    print(f"\n{'generator':12s}{'songs':>6s}" + "".join(f"{c:>11s}" for c in cols))
    rows = []
    for g in gen_list:
        s = labels == g
        r = [cv["clf"][s].mean(), logo[g][0], cv["comb16"][s].mean(), cv["comb44"][s].mean(), cv["comb2"][s].mean(),
             cv["full"][s].mean(), logo[g][1]]
        rows.append(r)
        print(f"{g:12s}{n[g]:6d}" + "".join(f"{100 * v:10.0f}%" for v in r))
    w = np.array([n[g] for g in gen_list])
    print(f"{'ALL':12s}{w.sum():6d}" + "".join(f"{100 * v:10.0f}%" for v in (w @ np.array(rows)) / w.sum()))
    print(f"{'REAL flagged':12s}{int(real.sum()):6d}" + "".join(
        f"{100 * v:10.1f}%" for v in (cv["clf"][real].mean(), np.mean([f[0] for f in fa]), cv["comb16"][real].mean(),
                                      cv["comb44"][real].mean(), cv["comb2"][real].mean(), cv["full"][real].mean(),
                                      np.mean([f[1] for f in fa]))))
    print("(REAL flagged: cross-validated columns are 5% by design; leave-one-out columns are measured on real\n"
          " songs not used to set the cut-off.)")


if __name__ == "__main__":
    main()

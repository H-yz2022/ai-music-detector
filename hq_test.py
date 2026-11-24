r"""16 kHz (1-8 kHz band) vs 44.1 kHz (5-16 kHz band, the paper's main setting), on the same songs.

Songs: the 7,894 real FMA clips and the 4,463 Echoes songs (12 generators), first 30 s.
SONICS is left out because it only exists at 16 kHz. At 44.1 kHz every song was
first re-encoded to the same 192 kbps MP3 (features_hq\, from build_fakeprints.py
--spectrum); at 16 kHz to the same 40 kbps MP3 (features\).

Some generators make audio at a low sample rate (nothing above 8 kHz, say). In the
5-16 kHz band that alone gives them away, which is not the decoder artifact. So the
script first measures how far up each source has content, and also reports the
results for full-band sources only.

For each setup (all cut-offs flag 5% of real songs; real clips grouped by artist):
- classifier, cross-validated (all 12 generators in training);
- classifier, leave one generator out (like logo_test.py);
- precise comb (training-free).

  python -I hq_test.py
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold  # noqa: E402

import fakeprint as fpm  # noqa: E402
from cutoff_test import artists_of  # noqa: E402
from detect import SHARPNESS_THRESHOLD  # noqa: E402
from logo_test import fit  # noqa: E402

HERE = Path(__file__).resolve().parent


def load(paths, kind):
    """(freqs, fakeprints, gens, names[, coarse spectra, coarse band edges]) from .npz files."""
    fps, gens, names, coarse, freqs, edges = [], [], [], [], None, None
    for p in paths:
        d = np.load(p)
        freqs = d["freqs"]
        if kind == "spec":  # 44.1 kHz files keep the band spectrum: make the fakeprints here
            fps.append(np.stack([fpm.fakeprint(freqs, s.astype(np.float64), float(d["fmin"]), float(d["fmax"]))[1]
                                 for s in d["spec"]]).astype(np.float32))
            coarse.append(d["coarse"])
            edges = d["coarse_edges"]
        else:
            fps.append(d["fp"].astype(np.float32))
        gens += list(d["gens"])
        names += list(d["names"])
    out = (freqs, np.concatenate(fps), np.array(gens), np.array(names))
    return out + ((np.concatenate(coarse), edges) if kind == "spec" else ())


def _precise_part(job):  # worker process
    freqs, fps = job
    return [fpm.precise_comb(freqs, fp.astype(np.float64))[0] for fp in fps]


def precise_all(freqs, X, workers=3, chunk=400):
    with ProcessPoolExecutor(workers) as pool:
        jobs = [(freqs, X[i:i + chunk]) for i in range(0, len(X), chunk)]
        return np.array([s for part in pool.map(_precise_part, jobs) for s in part])


def content_top(coarse, edges, drop_db=30.0):
    """Highest 500 Hz band whose level is within drop_db of the 5-8 kHz level, per song (kHz)."""
    lo = edges[:-1]
    ref = coarse[:, (lo >= 5000) & (lo < 8000)].mean(axis=1, keepdims=True)
    ok = coarse > ref - drop_db
    top = np.array([lo[np.flatnonzero(r)[-1]] + 500 if r.any() else 0 for r in ok])
    return top / 1000


def evaluate(X, labels, real_groups, echo_groups, gen_list):
    """Cross-validated and leave-one-generator-out catch rates per generator at 5% real false alarms."""
    real = labels == "real"
    y = ~real
    groups = np.where(real, real_groups, echo_groups)
    oof = np.zeros(len(y))
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, labels, groups), 1):
        use = np.ones(len(y), bool)
        use[te] = False
        oof[te] = fit(X, y, use).predict_proba(X[te])[:, 1]
        print(f"    cv fold {k}")
    cut = np.quantile(oof[real], 0.95)
    cv = {g: np.mean(oof[labels == g] > cut) for g in gen_list}

    real_idx = np.flatnonzero(real)
    fold = np.empty(len(real_idx), int)
    for k, (_, te) in enumerate(GroupKFold(len(gen_list)).split(real_idx, groups=real_groups[real])):
        fold[te] = k
    logo = {}
    for gi, g in enumerate(gen_list):
        cal = real_idx[fold == gi]
        use = labels != g
        use[cal] = False
        m = fit(X, y, use)
        c = np.quantile(m.predict_proba(X[cal])[:, 1], 0.95)
        logo[g] = np.mean(m.predict_proba(X[labels == g])[:, 1] > c)
        print(f"    left out {g}")
    return cv, logo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hq", nargs="+", default=[str(HERE / "features_hq" / "fma_small.npz"),
                                                str(HERE / "features_hq" / "echoes.npz")])
    ap.add_argument("--lq", nargs="+", default=[str(HERE / "features" / "fma_small.npz"),
                                                str(HERE / "features" / "unseen" / "echoes.npz")])
    ap.add_argument("--artists", default=str(HERE / "features" / "fma_artists.csv"))
    ap.add_argument("--full-band-khz", type=float, default=16.0)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    print("Loading 44.1 kHz spectra (making fakeprints) and 16 kHz fakeprints ...")
    fh, Xh, gh, nh, coarse, edges = load(args.hq, "spec")
    fl, Xl, gl, nl = load(args.lq, "fp")
    common = sorted(set(nh) & set(nl))  # same songs in both setups
    ih = {n: i for i, n in enumerate(nh)}
    il = {n: i for i, n in enumerate(nl)}
    sel_h, sel_l = np.array([ih[n] for n in common]), np.array([il[n] for n in common])
    Xh, Xl, labels, coarse = Xh[sel_h], Xl[sel_l], gh[sel_h], coarse[sel_h]
    names = np.array(common)
    real = labels == "real"
    gen_list = sorted(set(labels[~real]))
    print(f"  {len(names)} songs in both setups: {int(real.sum())} real, {int((~real).sum())} AI")

    top = content_top(coarse, edges)
    print(f"\nHow far up each source has content (44.1 kHz files, median of songs; "
          f"'full band' = up to {args.full_band_khz:g} kHz or more):")
    full_band = {}
    for g in ["real"] + gen_list:
        t = top[labels == g]
        full_band[g] = np.median(t) >= args.full_band_khz
        print(f"  {g:12s} {np.median(t):5.1f} kHz   ({100 * np.mean(t >= args.full_band_khz):.0f}% of songs full band)")

    art = artists_of(names, labels, args.artists)
    real_groups = np.array([f"artist{a}" if a is not None else n for a, n in zip(art, names)], dtype=object)
    echo_groups = names.astype(object)

    print("\nPrecise comb (training-free) ...")
    sh = precise_all(fh, Xh, args.workers)
    cut_h = np.quantile(sh[real], 0.95)
    sl = precise_all(fl, Xl, args.workers)
    print(f"  cut-offs (5% of real songs): 16 kHz {np.quantile(sl[real], 0.95):.2f} "
          f"(detect.py uses {SHARPNESS_THRESHOLD}), 44.1 kHz {cut_h:.2f}")
    cut_l = np.quantile(sl[real], 0.95)

    res = {}
    for tag, X in (("16k", Xl), ("44k", Xh)):
        print(f"\nClassifier, {tag} ...")
        res[tag] = evaluate(X, labels, real_groups, echo_groups, gen_list)

    n = {g: int((labels == g).sum()) for g in gen_list}
    print(f"\n{'generator':12s}{'songs':>6s}{'band':>6s} | {'cross-validated':>17s} | {'leave one out':>15s} | "
          f"{'precise comb':>14s}")
    print(f"{'':12s}{'':6s}{'':6s} | {'16k':>8s}{'44.1k':>9s} | {'16k':>7s}{'44.1k':>8s} | {'16k':>7s}{'44.1k':>7s}")
    rows = []
    for g in gen_list:
        r = (res["16k"][0][g], res["44k"][0][g], res["16k"][1][g], res["44k"][1][g],
             np.mean(sl[labels == g] > cut_l), np.mean(sh[labels == g] > cut_h))
        rows.append((g, r))
        print(f"{g:12s}{n[g]:6d}{'full' if full_band[g] else 'low':>6s} | {100 * r[0]:7.0f}%{100 * r[1]:8.0f}% | "
              f"{100 * r[2]:6.0f}%{100 * r[3]:7.0f}% | {100 * r[4]:6.0f}%{100 * r[5]:6.0f}%")
    for title, keep in (("ALL", lambda g: True), ("FULL BAND", lambda g: full_band[g])):
        gs = [(g, r) for g, r in rows if keep(g)]
        if not gs:
            print(f"{title:12s}  (none)")
            continue
        w = np.array([n[g] for g, _ in gs], dtype=int)
        avg = [100 * np.sum(w * np.array([r[i] for _, r in gs])) / w.sum() for i in range(6)]
        print(f"{title:12s}{w.sum():6d}{'':6s} | {avg[0]:7.0f}%{avg[1]:8.0f}% | {avg[2]:6.0f}%{avg[3]:7.0f}% | "
              f"{avg[4]:6.0f}%{avg[5]:6.0f}%")
    print("(All cut-offs flag 5% of real songs. SONICS is not included, so numbers differ from earlier tables.)")


if __name__ == "__main__":
    main()

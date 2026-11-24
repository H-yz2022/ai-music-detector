r"""One boundary per generator family: does it win back Udio without losing the others?

The final model (train_final.py) is one yes/no logistic regression: a single
boundary for all 12 generator families. Here the classifier is multi-class
(real + one class per family: suno, udio, acestep, ...), with exactly the same
song weights (real songs half, AI songs half, each family in proportion to its
songs). So the only change is that each family gets its own boundary.
AI score = 1 - P(real). Settings fixed before running.

  python -I family_test.py --phase cv     # cross-validation + fresh songs, same folds as train_final.py
  python -I family_test.py --phase logo   # leave one generator out, same setup as logo_test.py
  python -I family_test.py --phase save   # train on everything -> model_family.joblib
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold  # noqa: E402

import fakeprint as fpm  # noqa: E402
from build_fakeprints import song_key  # noqa: E402
from cutoff_test import artists_of  # noqa: E402
from detect import SHARPNESS_THRESHOLD  # noqa: E402
from scale_test import load_features, load_test  # noqa: E402

HERE = Path(__file__).resolve().parent


def fit_family(X, labels, use):
    """Multi-class logistic regression on rows where `use` is True (others get weight 0 and label 'real')."""
    lab = np.where(use, labels, "real")
    ai = lab != "real"
    w = np.zeros(len(lab))
    for cls in (False, True):
        sel = use & (ai == cls)
        w[sel] = use.sum() / (2 * sel.sum())
    t = time.time()
    m = LogisticRegression(max_iter=5000).fit(X, lab, sample_weight=w)
    print(f"    fit: {len(m.classes_)} classes, {(time.time() - t) / 60:.1f} min")
    return m


def ai_score(model, X):
    return 1 - model.predict_proba(X)[:, list(model.classes_).index("real")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["cv", "logo", "save"], default="cv")
    ap.add_argument("--features", default=str(HERE / "features"))
    ap.add_argument("--unseen", default=str(HERE / "features" / "unseen" / "echoes.npz"))
    ap.add_argument("--artists", default=str(HERE / "features" / "fma_artists.csv"))
    ap.add_argument("--test", default=str(HERE / "data_fresh"))
    ap.add_argument("--out", default=str(HERE / "model_family.joblib"))
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--fmin", type=float, default=1000.0)
    ap.add_argument("--fmax", type=float, default=8000.0)
    ap.add_argument("--max-duration", type=float, default=30.0)
    args = ap.parse_args()

    freqs, Xp, gens, names = load_features(args.features, args)
    u = np.load(args.unseen)
    X = np.concatenate([Xp, u["fp"].astype(np.float32)])
    labels, allnames = np.concatenate([gens, u["gens"]]), np.concatenate([names, u["names"]])
    is_echo = np.r_[np.zeros(len(gens), bool), np.ones(len(u["gens"]), bool)]
    del Xp
    real = labels == "real"
    art = artists_of(allnames, labels, args.artists)
    _, Xt, tgens, _ = load_test(Path(args.test), args)

    if args.phase in ("cv", "save"):
        # Same groups and folds as train_final.py, so the numbers compare one to one.
        groups = np.array([f"artist{a}" if a is not None else n if "/" in n else f"{g}:{song_key(n)}"
                           for a, g, n in zip(art, labels, allnames)])
        oof = np.zeros(len(labels))
        if args.phase == "cv":
            for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0)
                                        .split(X, labels, groups), 1):
                use = np.ones(len(labels), bool)
                use[te] = False
                oof[te] = ai_score(fit_family(X, labels, use), X[te])
                print(f"  fold {k} done")
            cut = float(np.quantile(oof[real], 0.95))
            print(f"\nCut-off: AI score > {cut:.3f} flags 5% of real songs (cross-validated AUC "
                  f"{roc_auc_score(~real, oof):.3f})")
            print("Cross-validated catch rate per generator:")
            for g in sorted(set(labels[~real])):
                print(f"  {g:12s} {100 * np.mean(oof[labels == g] > cut):5.1f}%  ({(labels == g).sum()} songs)")
            np.save(Path(args.out).with_suffix(".cut.npy"), np.array([cut]))
        cut_file = Path(args.out).with_suffix(".cut.npy")
        if not cut_file.exists():
            sys.exit(f"{cut_file.name} not found: run --phase cv first (with the same --out).")
        cut = float(np.load(cut_file)[0])
        model = fit_family(X, labels, np.ones(len(labels), bool))
        p = ai_score(model, Xt)
        print("Fresh songs (never trained on):")
        for g in ("real", "suno", "udio"):
            rate = np.mean(p[tgens == g] <= cut) if g == "real" else np.mean(p[tgens == g] > cut)
            print(f"  {g:5s} {'called real' if g == 'real' else 'caught':12s} {100 * rate:.0f}%")
        print(f"  AUC {roc_auc_score(tgens != 'real', p):.3f}")
        if args.phase == "save":
            joblib.dump({"model": model, "sr": args.sr, "fmin": args.fmin, "fmax": args.fmax,
                         "max_duration": args.max_duration, "n_features": X.shape[1], "cutoff": cut,
                         "kind": "family"}, args.out)
            print(f"Saved {args.out}")
        return

    # Leave one generator out (same as logo_test.py: holding out g removes every song labelled g).
    gen_list = sorted(set(u["gens"]))
    real_idx = np.flatnonzero(real)
    grp = np.array([a if a is not None else f"track{i}" for i, a in enumerate(art[real])], dtype=object)
    real_fold = np.empty(len(real_idx), int)
    for k, (_, te) in enumerate(GroupKFold(len(gen_list)).split(real_idx, groups=grp)):
        real_fold[te] = k
    cache = Path(args.unseen).with_name(Path(args.unseen).stem + "_precise.npy")
    comb_u = np.load(cache) > SHARPNESS_THRESHOLD
    Xr = Xt[tgens == "real"]
    comb_r = np.array([fpm.precise_comb(freqs, fp.astype(np.float64))[0] for fp in Xr]) > SHARPNESS_THRESHOLD
    rows = {}
    for gi, g in enumerate(gen_list):
        print(f"  holding out {g} ...")
        cal = real_idx[real_fold == gi]
        use = labels != g
        use[cal] = False
        m = fit_family(X, labels, use)
        cut = np.quantile(ai_score(m, X[cal]), 0.95)
        sel = is_echo & (labels == g)
        hit = ai_score(m, X[sel]) > cut
        fr = ai_score(m, Xr) > cut
        rows[g] = (sel.sum(), hit.mean(), (hit | comb_u[labels[is_echo] == g]).mean(), fr.mean(), (fr | comb_r).mean())
        print(f"  {g:12s} caught {100 * hit.mean():.0f}%, with precise comb {100 * rows[g][2]:.0f}%")
    n = np.array([r[0] for r in rows.values()])
    print(f"\n{'generator':14s}{'songs':>6s}{'family model':>14s}{'+ precise comb':>16s}")
    for g, r in rows.items():
        print(f"{g:14s}{r[0]:6d}{100 * r[1]:13.0f}%{100 * r[2]:15.0f}%")
    print(f"{'ALL':14s}{n.sum():6d}{100 * np.sum(n * [r[1] for r in rows.values()]) / n.sum():13.0f}%"
          f"{100 * np.sum(n * [r[2] for r in rows.values()]) / n.sum():15.0f}%")
    print(f"fresh real flagged (mean over folds): {100 * np.mean([r[3] for r in rows.values()]):.1f}%, "
          f"with precise comb {100 * np.mean([r[4] for r in rows.values()]):.1f}%")


if __name__ == "__main__":
    main()

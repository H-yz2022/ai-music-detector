r"""Non-linear classifiers on the same two-band fakeprints, compared with logistic regression.

Logistic regression draws one straight boundary (or one per family). The other models can draw a bent
one, e.g. "peak at bin A AND peak at bin B": LightGBM and a random forest (decision trees), a small
neural network (MLP), and an RBF SVM. The risk: they may learn the exact peaks of the training
generators (or the look of the training real songs) and miss new ones. So three tests, all at 5% false alarms:
1. cv:    5-fold cross-validation (real songs grouped by artist); every generator in training;
2. logo:  leave one generator family out (Udio merged); that family is never seen in training.
          The held-out real artist fold is split in two: one half sets the cut-off, the other
          half measures false alarms;
3. fresh: fresh44 (728 new songs: Suno V5.5, MiniMax never seen, Lyria 3, new real songs), with
          the final model and the cut-off from cross-validation (as train_twoband.py does).
Songs: the same as train_twoband.py --extra-dir features_hq\more44 (12,357 + 2,762).
"joint" = classifier or precise comb (either band), one cut-off on the combined score.
Model settings are fixed before running (no tuning on these tests).

  python -I nonlinear_test.py --quick 3000 --tests cv          # small sample: timing only
  python -I nonlinear_test.py --models lr family lgbm            # the full comparison
  python -I nonlinear_test.py --extra-real features_hq\more_real --hold-real etree --tests cv fresh \
      --out reports\nonlinear_morereal_holdetree.json             # + varied real songs (more_real.py), live
                                                                  #   concerts kept out to test on
  python -I nonlinear_test.py --extra-real features_hq\more_real --out reports\nonlinear_morereal.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold  # noqa: E402

from detect import joint_score  # noqa: E402

HERE = Path(__file__).resolve().parent
LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                   colsample_bytree=0.3, subsample=0.8, subsample_freq=1, reg_lambda=1.0, max_bin=63,
                   random_state=0, n_jobs=10, verbose=-1)
MLP_PARAMS = dict(hidden_layer_sizes=(256,), alpha=1e-3, batch_size=256, learning_rate_init=1e-3, max_iter=100,
                  early_stopping=True, validation_fraction=0.1, n_iter_no_change=5, random_state=0)
RF_PARAMS = dict(n_estimators=500, max_features="sqrt", min_samples_leaf=2, n_jobs=10, random_state=0)
SVM_COMPONENTS = 2000  # RBF kernel approximated with 2,000 Nystroem features, then logistic regression
NAMES = {"lr": "LR one boundary", "family": "LR per family", "lgbm": "LightGBM",
         "mlp": "Neural network (MLP, 256 hidden)", "rf": "Random forest", "svm": "RBF SVM (kernel approx.)"}


def balanced(y, use):
    w = np.zeros(len(y))
    for cls in (False, True):
        sel = use & (y == cls)
        w[sel] = use.sum() / (2 * sel.sum())
    return w


def train(kind, X, y, fam, use):
    """Fit model `kind` on rows `use`; returns a function: features -> P(AI)."""
    if kind == "lr":
        from logo_test import fit
        m = fit(X, y, use)
        return lambda Xs: m.predict_proba(Xs)[:, 1]
    if kind == "family":
        from family_test import ai_score, fit_family
        m = fit_family(X, fam, use)
        return lambda Xs: ai_score(m, Xs)
    if kind == "lgbm":
        from lightgbm import LGBMClassifier
        idx = np.flatnonzero(use)
        m = LGBMClassifier(**LGBM_PARAMS).fit(X[idx], y[idx], sample_weight=balanced(y, use)[idx])
        return lambda Xs: m.predict_proba(Xs)[:, 1]
    idx = np.flatnonzero(use)
    w = balanced(y, use)[idx]
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        m = MLPClassifier(**MLP_PARAMS).fit(X[idx], y[idx], sample_weight=w)
    elif kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(**RF_PARAMS).fit(X[idx], y[idx], sample_weight=w)
    elif kind == "svm":
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        gamma = 1.0 / (X.shape[1] * float(X[idx[::10]].var()))  # sklearn's 'scale' rule
        m = make_pipeline(Nystroem(gamma=gamma, n_components=SVM_COMPONENTS, random_state=0),
                          LogisticRegression(max_iter=5000))
        m.fit(X[idx], y[idx], logisticregression__sample_weight=w)
    else:
        raise ValueError(kind)
    return lambda Xs: m.predict_proba(Xs)[:, 1]


def pct(ref, x):
    return np.searchsorted(np.sort(ref), x, side="right") / len(ref)


def load_all(quick, seed=0, extra_real=None, hold=()):
    """Training songs: (X, labels, groups, s16, s44, fam, origin, held). extra_real: folder of more real songs
    (more_real.py); songs of the `hold` origins are kept out of training and returned in `held` for testing."""
    from more44_test import FAMILY, load_new, load_old
    print("Loading the songs ...")
    X, labels, groups, s16, s44 = load_old()
    origin = np.where(labels == "real", "fma_small", "ai")
    new = load_new(HERE / "features_hq" / "more44")
    parts = [new] + ([dict(np.load(p)) for p in sorted(Path(extra_real).glob("*.npz"))] if extra_real else [])
    for p in parts:
        if "origin" in p:  # extra real songs: only files at 44.1 kHz or more (what the two-band model gets)
            ok = p["src_sr"] >= 44100
            p = {k: (v[ok] if getattr(v, "ndim", 0) and len(v) == len(ok) else v) for k, v in p.items()}
        X = np.vstack([X, np.hstack([p["fp0"], p["fp1"]]).astype(np.float32)])
        labels = np.concatenate([labels, p["sources"]]).astype(str)
        groups = np.concatenate([groups, p["groups"].astype(object)])
        s16, s44 = np.concatenate([s16, p["sharp0"]]), np.concatenate([s44, p["sharp1"]])
        origin = np.concatenate([origin, p["origin"] if "origin" in p else np.full(len(p["sources"]), "ai")])
    held = {}
    for h in hold:
        m = origin == h
        held[h] = {"X": X[m], "s16": s16[m], "s44": s44[m]}
        X, labels, groups, s16, s44, origin = X[~m], labels[~m], groups[~m], s16[~m], s44[~m], origin[~m]
    if quick:  # random subset, same share of each source
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(labels), quick, replace=False))
        X, labels, groups, s16, s44, origin = X[keep], labels[keep], groups[keep], s16[keep], s44[keep], origin[keep]
    fam = np.array([FAMILY.get(s, s) for s in labels])
    print(f"  {len(labels)} songs, {X.shape[1]} features: {int((labels == 'real').sum())} real ("
          + ", ".join(f"{o} {int((origin == o).sum())}" for o in sorted(set(origin) - {"ai"}))
          + f"), {int((labels != 'real').sum())} AI from {len(set(fam)) - 1} families"
          + (f"; held out of training: " + ", ".join(f"{h} {len(v['X'])}" for h, v in held.items()) if held else ""))
    return X, labels, groups, s16, s44, fam, origin, held


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["lr", "family", "lgbm"], choices=list(NAMES))
    ap.add_argument("--tests", nargs="+", default=["cv", "fresh", "logo"], choices=["cv", "fresh", "logo"])
    ap.add_argument("--quick", type=int, default=0, help="use only this many random songs (timing check)")
    ap.add_argument("--fresh", default=str(HERE / "features_hq" / "unseen" / "fresh44.npz"))
    ap.add_argument("--out", default=str(HERE / "reports" / "nonlinear_test.json"))
    ap.add_argument("--extra-real", default=None, help="folder of more real songs (more_real.py) to train on")
    ap.add_argument("--hold-real", nargs="*", default=[], help="origins of extra real songs to keep out of "
                    "training and test on (e.g. etree)")
    args = ap.parse_args()

    X, labels, groups, s16, s44, fam, origin, held = load_all(args.quick, extra_real=args.extra_real,
                                                             hold=args.hold_real)
    real, y = labels == "real", labels != "real"
    results = json.loads(Path(args.out).read_text()) if Path(args.out).exists() and not args.quick else {}

    for kind in args.models:
        res = results.setdefault(kind, {})
        print(f"\n===== {NAMES[kind]} =====")
        if "cv" in args.tests or "fresh" in args.tests:
            t = time.time()
            oof = np.zeros(len(y))
            for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, labels, groups), 1):
                use = np.ones(len(y), bool)
                use[te] = False
                oof[te] = train(kind, X, y, fam, use)(X[te])
                print(f"  cv fold {k} ({(time.time() - t) / 60:.1f} min)")
            cut = float(np.quantile(oof[real], 0.95))
            refs = {"clf": np.sort(oof[real]), "s16": np.sort(s16[real]), "s44": np.sort(s44[real])}
            joint = joint_score(refs, oof, s16, s44)
            jcut = float(np.quantile(joint[real], 0.95))
            res["cv"] = {g: [float(np.mean(oof[labels == g] > cut)), float(np.mean(joint[labels == g] > jcut))]
                         for g in sorted(set(labels[y]))}
            res["cv"]["ALL AI"] = [float(np.mean(oof[y] > cut)), float(np.mean(joint[y] > jcut))]
            for o in sorted(set(origin[real])):  # false alarms by where the real songs come from
                m = real & (origin == o)
                res["cv"][f"real {o}"] = [float(np.mean(oof[m] > cut)), float(np.mean(joint[m] > jcut))]
            print(f"  cross-validated catch rate (cut-off {cut:.3f}): classifier / joint (real rows: flagged)")
            for g, (a, b) in res["cv"].items():
                print(f"    {g:12s} {100 * a:5.1f}%  {100 * b:5.1f}%")
        if "fresh" in args.tests:
            t = time.time()
            score = train(kind, X, y, fam, np.ones(len(y), bool))
            f = np.load(args.fresh)
            hi = f["src_sr"] >= 44100  # predict uses the two-band model for these files
            src = f["sources"].astype(str)
            p = score(np.hstack([f["fp0"], f["fp1"]]))
            flag, jflag = p > cut, joint_score(refs, p, f["sharp0"], f["sharp1"]) > jcut
            rows = {s: (src == s) & hi for s in sorted(set(src))}
            rows["ALL REAL"] = np.char.startswith(src, "real") & hi
            res["fresh"] = {s: [float(flag[m].mean()), float(jflag[m].mean()), int(m.sum())] for s, m in rows.items()}
            for h, v in held.items():  # real songs from a source kept out of training
                ph = score(v["X"])
                res["fresh"][f"held {h}"] = [float(np.mean(ph > cut)),
                                             float(np.mean(joint_score(refs, ph, v["s16"], v["s44"]) > jcut)),
                                             int(len(ph))]
            print(f"  fresh44 (final model {(time.time() - t) / 60:.1f} min; real rows: share flagged, AI rows: caught)")
            for s, (a, b, n) in res["fresh"].items():
                print(f"    {s:14s}{n:5d} {100 * a:5.1f}%  {100 * b:5.1f}%")
        if "logo" in args.tests:
            t = time.time()
            fams = sorted(set(fam[y]))
            real_idx = np.flatnonzero(real)
            fold = np.empty(len(real_idx), int)
            for k, (_, te) in enumerate(GroupKFold(len(fams)).split(real_idx, groups=groups[real])):
                fold[te] = k
            rng = np.random.default_rng(0)
            res["logo"], fa = {}, []
            print("  leave one family out: classifier / joint (never trained on this family)")
            for i, g in enumerate(fams):
                cal = rng.permutation(real_idx[fold == i])
                set_cut, check = cal[: len(cal) // 2], cal[len(cal) // 2:]
                use = fam != g
                use[cal] = False
                score = train(kind, X, y, fam, use)
                gi = np.flatnonzero(fam == g)
                ps, pg, pc = score(X[set_cut]), score(X[gi]), score(X[check])
                comb = lambda pr, ii: np.maximum.reduce([pct(ps, pr), pct(s16[set_cut], s16[ii]),  # noqa: E731
                                                         pct(s44[set_cut], s44[ii])])
                c1, c2 = np.quantile(ps, 0.95), np.quantile(comb(ps, set_cut), 0.95)
                res["logo"][g] = [float(np.mean(pg > c1)), float(np.mean(comb(pg, gi) > c2)), int(len(gi))]
                fa.append([float(np.mean(pc > c1)), float(np.mean(comb(pc, check) > c2))])
                print(f"    {g:12s} {100 * res['logo'][g][0]:5.1f}%  {100 * res['logo'][g][1]:5.1f}%  "
                      f"({(time.time() - t) / 60:.1f} min)")
            n = np.array([v[2] for v in res["logo"].values()])
            res["logo"]["ALL AI"] = [float(n @ [v[0] for v in res["logo"].values()] / n.sum()),
                                     float(n @ [v[1] for v in res["logo"].values()] / n.sum()), int(n.sum())]
            res["logo"]["REAL flagged"] = [float(np.mean([a[0] for a in fa])), float(np.mean([a[1] for a in fa])),
                                           int(len(real_idx))]
            for g in ("ALL AI", "REAL flagged"):
                print(f"    {g:12s} {100 * res['logo'][g][0]:5.1f}%  {100 * res['logo'][g][1]:5.1f}%")
        if not args.quick:
            Path(args.out).write_text(json.dumps(results, indent=1))
            print(f"  saved {args.out}")


if __name__ == "__main__":
    main()

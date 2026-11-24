r"""Train the final classifier for `detect.py predict` on every song we have.

Training data (fakeprints in features\): 7,894 real FMA clips, 4,074 SONICS
songs (Suno v2-v3.5, Udio) and 4,463 Echoes songs from 12 more generators.
Setup: 16 kHz, 1-8 kHz band, first 30 s, 40 kbps MP3.

The cut-off comes from 5-fold cross-validation: every song gets P(AI) from a
model that never saw it (real clips grouped by artist, so an artist never sits
on both sides), and the cut-off flags 5% of real songs. The 200 fresh songs in
data_fresh\ are never trained on and give a last check.

  python -I train_final.py            # writes model_final.joblib
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from build_fakeprints import song_key  # noqa: E402
from cutoff_test import artists_of  # noqa: E402
from logo_test import fit  # noqa: E402
from scale_test import load_features, load_test  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=str(HERE / "features"))
    ap.add_argument("--extra", nargs="*", default=[str(HERE / "features" / "unseen" / "echoes.npz")])
    ap.add_argument("--artists", default=str(HERE / "features" / "fma_artists.csv"))
    ap.add_argument("--test", default=str(HERE / "data_fresh"))
    ap.add_argument("--false-alarms", type=float, default=5.0)
    ap.add_argument("--out", default=str(HERE / "model_final.joblib"))
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--fmin", type=float, default=1000.0)
    ap.add_argument("--fmax", type=float, default=8000.0)
    ap.add_argument("--max-duration", type=float, default=30.0)
    args = ap.parse_args()

    print("Training songs:")
    freqs, Xp, gens, names = load_features(args.features, args)
    parts, gen_parts, name_parts = [Xp], [gens], [names]
    for path in args.extra:
        d = np.load(path)
        if not np.array_equal(freqs, d["freqs"]):
            sys.exit(f"{path} has different frequency bins.")
        parts.append(d["fp"].astype(np.float32))
        gen_parts.append(d["gens"])
        name_parts.append(d["names"])
        print(f"  {Path(path).name}: {len(d['gens'])} songs")
    X, gens, names = np.concatenate(parts), np.concatenate(gen_parts), np.concatenate(name_parts)
    del parts, Xp
    y = gens != "real"
    # Groups for cross-validation: real clips by artist, SONICS by prompt (its two sister songs),
    # Echoes songs one by one (their paths contain a "/").
    art = artists_of(names, gens, args.artists)
    groups = np.array([f"artist{a}" if a is not None else n if "/" in n else f"{g}:{song_key(n)}"
                       for a, g, n in zip(art, gens, names)])
    print(f"  total {len(y)}: {int((~y).sum())} real, {int(y.sum())} AI from {len(set(gens[y]))} generators")

    print("Cross-validation for the cut-off (5 folds) ...")
    oof = np.zeros(len(y))
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, gens, groups), 1):
        use = np.ones(len(y), bool)
        use[te] = False
        oof[te] = fit(X, y, use).predict_proba(X[te])[:, 1]
        print(f"  fold {k} done")
    cut = float(np.quantile(oof[~y], 1 - args.false_alarms / 100))
    print(f"\nCut-off: P(AI) > {cut:.3f} flags {args.false_alarms:g}% of real songs (cross-validated AUC "
          f"{roc_auc_score(y, oof):.3f})")
    print("Cross-validated catch rate per generator (these generators are in training):")
    for g in sorted(set(gens[y])):
        print(f"  {g:12s} {100 * np.mean(oof[gens == g] > cut):5.1f}%  ({(gens == g).sum()} songs)")

    print("\nFinal model on all songs ...")
    model = fit(X, y, np.ones(len(y), bool))
    joblib.dump({"model": model, "sr": args.sr, "fmin": args.fmin, "fmax": args.fmax,
                 "max_duration": args.max_duration, "n_features": X.shape[1], "cutoff": cut,
                 "trained_on": {g: int((gens == g).sum()) for g in sorted(set(gens))}}, args.out)
    print(f"Saved {args.out}")

    print("Fresh songs (never trained on):")
    _, Xt, tgens, _ = load_test(Path(args.test), args)
    p = model.predict_proba(Xt)[:, 1]
    for g in ("real", "suno", "udio"):
        rate = np.mean(p[tgens == g] > cut) if g != "real" else np.mean(p[tgens == g] <= cut)
        print(f"  {g:5s} {'called real' if g == 'real' else 'caught':12s} {100 * rate:.0f}%  ({(tgens == g).sum()} songs)")
    print(f"  AUC {roc_auc_score(tgens != 'real', p):.3f}")


if __name__ == "__main__":
    main()

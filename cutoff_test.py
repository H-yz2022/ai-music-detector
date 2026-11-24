r"""Choose the classifier's cut-off on the training data, then test it on fresh songs.

The default cut-off (call a song AI when P(AI) >= 0.5) is not tuned for anything.
Here the cut-off comes from 5-fold cross-validation on the training pool
(features\*.npz): every training song gets P(AI) from a model that never saw it
(sister songs from the same prompt stay in the same fold), and the cut-off is
set so that a fixed share of real training songs is flagged (5% and 1% by
default, decided before looking at the fresh songs). Then a model trained on
the whole pool is scored on data_fresh\ with those cut-offs.

--artist-disjoint (needs features\fma_artists.csv from `fetch_data.py fma-artists`)
drops every training clip by an artist who also appears in the fresh real
songs, and groups the cross-validation folds by artist. If the results stay the
same, the model wasn't just recognising artists it had heard before.

  python -I cutoff_test.py
  python -I cutoff_test.py --artist-disjoint
"""
import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from build_fakeprints import song_key  # noqa: E402
from detect import make_model  # noqa: E402
from scale_test import load_features, load_test  # noqa: E402

HERE = Path(__file__).resolve().parent


def rates(p, gens, cut):
    """(% real called real, % Suno caught, % Udio caught) when P(AI) > cut means AI."""
    return (100 * np.mean(p[gens == "real"] <= cut), 100 * np.mean(p[gens == "suno"] > cut),
            100 * np.mean(p[gens == "udio"] > cut))


def default_groups(names, gens):
    """Cross-validation groups: one per FMA track, one per SONICS prompt (its two sister songs)."""
    return np.array([("fma" if g == "real" else "sonics") + song_key(n) for g, n in zip(gens, names)])


def cross_validated(X, gens, groups, folds=5):
    """P(AI) for every training song from a model that never saw it, or any song in its group."""
    y = gens != "real"
    oof = np.zeros(len(y))
    for k, (tr, te) in enumerate(StratifiedGroupKFold(folds, shuffle=True, random_state=0)
                                 .split(X, gens, groups), 1):
        oof[te] = make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        print(f"  fold {k} done")
    return oof


def artists_of(names, gens, path):
    """Artist id for each real FMA clip (None for AI songs or unknown tracks)."""
    with open(path, newline="") as f:
        table = {int(r["track_id"]): r["artist_id"] for r in csv.DictReader(f)}
    return np.array([table.get(int(song_key(n))) if g == "real" else None for n, g in zip(names, gens)], dtype=object)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=str(HERE / "features"))
    ap.add_argument("--test", default=str(HERE / "data_fresh"))
    ap.add_argument("--false-alarms", type=float, nargs="+", default=[5.0, 1.0],
                    help="%% of real training songs the cut-off may flag")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--artist-disjoint", action="store_true")
    ap.add_argument("--artists", default=str(HERE / "features" / "fma_artists.csv"))
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--fmin", type=float, default=1000.0)
    ap.add_argument("--fmax", type=float, default=8000.0)
    ap.add_argument("--max-duration", type=float, default=30.0)
    args = ap.parse_args()

    print("Training pool:")
    freqs, X, gens, names = load_features(args.features, args)
    print("Test songs (data_fresh):")
    tfreqs, Xt, tgens, tnames = load_test(Path(args.test), args)
    if not np.array_equal(freqs, tfreqs):
        sys.exit("Test fakeprints have different frequency bins from the training pool.")

    groups = default_groups(names, gens)
    if Path(args.artists).exists():
        art, tart = artists_of(names, gens, args.artists), artists_of(tnames, tgens, args.artists)
        test_artists = set(tart[tgens == "real"]) - {None}
        pool_artists = set(art[gens == "real"]) - {None}
        shared = np.array([a in test_artists for a in art])
        n_test = sum(a in pool_artists for a in tart[tgens == "real"])
        print(f"\nArtist overlap: {n_test} of {(tgens == 'real').sum()} fresh real clips are by an artist who is also "
              f"in the training pool ({shared.sum()} training clips by those artists).")
        if args.artist_disjoint:
            keep = ~shared
            X, gens, names, art = X[keep], gens[keep], names[keep], art[keep]
            groups = np.array([f"artist{a}" if a is not None else g for a, g in zip(art, groups[keep])])
            print(f"Artist-disjoint: dropped those {shared.sum()} training clips; folds are grouped by artist.")
    elif args.artist_disjoint:
        sys.exit(f"--artist-disjoint needs {args.artists} (python -I fetch_data.py fma-artists).")
    y = gens != "real"

    print(f"\nCross-validation ({args.folds} folds) ...")
    oof = cross_validated(X, gens, groups, args.folds)
    cuts = {"0.5 (default)": 0.5}
    for a in args.false_alarms:
        cut = float(np.quantile(oof[gens == "real"], 1 - a / 100))
        cuts[f"{a:g}% rule ({cut:.3f})"] = cut
    p = make_model().fit(X, y).predict_proba(Xt)[:, 1]

    head = f"{'cut-off':22s}{'real called real':>18s}{'Suno caught':>13s}{'Udio caught':>13s}"
    for title, probs, g, auc in (
            (f"Training pool, cross-validated ({len(y)} songs)", oof, gens, roc_auc_score(y, oof)),
            (f"Fresh songs ({len(tgens)}, model trained on the whole pool)", p, tgens,
             roc_auc_score(tgens != "real", p))):
        print(f"\n{title}, AUC {auc:.3f}\n{head}")
        for label, cut in cuts.items():
            r, s, u = rates(probs, g, cut)
            print(f"{label:22s}{r:17.1f}%{s:12.1f}%{u:12.1f}%")


if __name__ == "__main__":
    main()

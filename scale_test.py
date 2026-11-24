r"""Does more training data help? Train the paper's classifier on growing amounts of
data from features\*.npz (see build_fakeprints.py) and test every model on the
same held-out songs in data_fresh\, which are never used for training.

  python -I scale_test.py
  python -I scale_test.py --sizes 200 1000 4000 0 --repeats 3

Each size is a number of training songs: half real, half AI, the AI half split
evenly between Suno and Udio. 0 means every song available. Small sizes are
drawn several times (--repeats) and averaged, because one draw is noisy.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from detect import featurize_folder, make_model  # noqa: E402

HERE = Path(__file__).resolve().parent


def load_features(folder, args):
    """Returns (freqs, fakeprints, generator per song, file name per song) from every .npz in folder."""
    fps, gens, names, freqs = [], [], [], None
    for path in sorted(Path(folder).glob("*.npz")):
        d = np.load(path)
        setup = (int(d["sr"]), float(d["fmin"]), float(d["fmax"]), float(d["max_duration"]))
        if setup != (args.sr, args.fmin, args.fmax, args.max_duration):
            sys.exit(f"{path.name} was built with sr/fmin/fmax/duration {setup}, not the settings asked for here.")
        if freqs is not None and not np.array_equal(freqs, d["freqs"]):
            sys.exit(f"{path.name} has different frequency bins.")
        freqs = d["freqs"]
        fps.append(d["fp"].astype(np.float32))
        gens += list(d["gens"])
        names += list(d["names"])
        print(f"  {path.name}: {len(d['gens'])} songs")
    return freqs, np.concatenate(fps), np.array(gens), np.array(names)


def load_test(root, args):
    band = (args.sr, args.fmin, args.fmax, args.max_duration)
    items = featurize_folder(root / "real", "real", *band) + featurize_folder(root / "ai", "ai", *band)
    gens = []
    for it in items:
        m = re.search(r"_(suno|udio)_", it["name"].lower())
        gens.append(m.group(1) if m else ("real" if it["group"] == "real" else "ai"))
    return (items[0]["freqs"], np.stack([it["fp"] for it in items]).astype(np.float32), np.array(gens),
            np.array([it["name"] for it in items]))


def pick(gens, size, rng):
    """Indices for a training set of `size` songs: half real, a quarter Suno, a quarter Udio."""
    if size == 0:
        return np.arange(len(gens))
    want = {"real": size // 2, "suno": size // 4, "udio": size - size // 2 - size // 4}
    idx = []
    for g, n in want.items():
        pool = np.flatnonzero(gens == g)
        if n > len(pool):
            return None
        idx.append(rng.choice(pool, n, replace=False))
    return np.concatenate(idx)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=str(HERE / "features"))
    ap.add_argument("--test", default=str(HERE / "data_fresh"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 500, 1000, 2000, 4000, 8000, 0])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--fmin", type=float, default=1000.0)
    ap.add_argument("--fmax", type=float, default=8000.0)
    ap.add_argument("--max-duration", type=float, default=30.0)
    args = ap.parse_args()

    print("Training pool:")
    freqs, X, gens, _ = load_features(args.features, args)
    print("  total:", {g: int((gens == g).sum()) for g in ("real", "suno", "udio")})
    print("Test songs (data_fresh):")
    tfreqs, Xt, tgens, _ = load_test(Path(args.test), args)
    if not np.array_equal(freqs, tfreqs):
        sys.exit("Test fakeprints have different frequency bins from the training pool.")
    is_ai = tgens != "real"

    print(f"\n{'train songs':>12s}{'real called real':>18s}{'Suno caught':>13s}{'Udio caught':>13s}"
          f"{'AUC all AI':>12s}{'AUC Udio':>10s}")
    for size in args.sizes:
        rows = []
        for r in range(1 if size == 0 else args.repeats):
            idx = pick(gens, size, np.random.default_rng(r))
            if idx is None:
                break
            model = make_model().fit(X[idx], gens[idx] != "real")
            p = model.predict_proba(Xt)[:, 1]
            udio = (tgens == "udio") | (tgens == "real")
            rows.append([100 * np.mean(p[tgens == "real"] < 0.5), 100 * np.mean(p[tgens == "suno"] >= 0.5),
                         100 * np.mean(p[tgens == "udio"] >= 0.5), roc_auc_score(is_ai, p),
                         roc_auc_score(tgens[udio] == "udio", p[udio])])
        if not rows:
            print(f"{size:12d}  (not enough songs)")
            continue
        m = np.mean(rows, axis=0)
        label = f"all {len(gens)}" if size == 0 else str(size)
        print(f"{label:>12s}{m[0]:17.0f}%{m[1]:12.0f}%{m[2]:12.0f}%{m[3]:12.3f}{m[4]:10.3f}"
              + (f"   (mean of {len(rows)})" if len(rows) > 1 else ""))


if __name__ == "__main__":
    main()

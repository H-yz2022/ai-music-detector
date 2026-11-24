r"""Step 2, before downloading more songs: would more songs of the weak generators help?

Mureka (50%), Lyria Pro 3 (43%) and Mubert (17%) are caught poorly even when the model trains on them
(autocorr_test.json, cv). If that is because there are too few of their songs, the share caught keeps rising as more
of their songs are added; if their trace is just weak in the fakeprint, it levels off.
Learning curve, no download (stored fakeprints, features_tm\, as autocorr_test.py): for each weak family, its songs
are split into 3 folds by group (artist / prompt). For each test fold, the model trains on every other song plus a
share (0, 1/3, 2/3, all) of the family's other two folds. The cut-off is 95% of 20% of the real songs, held out by
artist group and never trained on. Udio (89% when trained on) and Sonauto are controls.
Reported: the share of the test fold caught (mean over the 3 folds) and, in brackets, how many of the family's
songs were trained on.
Read before running: "more songs help" if going from 2/3 to all of the other folds still adds at least 3 points
(the curve has not levelled off).

  python -I learning_test.py --kind lr        # LR one boundary (the predict model), ~20 min
  python -I learning_test.py --kind family    # LR per family, ~1.5 hours
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from sklearn.model_selection import GroupKFold, GroupShuffleSplit  # noqa: E402

import temporal_test as tt  # noqa: E402
tt.SKIP |= {"tstd0", "tstd1", "tco0", "tco1", "shape0", "shape1", "stereo"}  # only the fakeprints are used

HERE = Path(__file__).resolve().parent
FAMILIES = ["mureka", "lyria", "mubert", "sonauto", "udio"]
SHARES = [0.0, 1 / 3, 2 / 3, 1.0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", default="lr", choices=["lr", "family"])
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--quick", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "reports" / "learning_test.json"))
    args = ap.parse_args()
    out = Path(args.out)
    results = json.loads(out.read_text()) if out.exists() and not args.quick else {}
    print("Loading ...", flush=True)
    d, _, _ = tt.load(HERE / "features_tm", args.quick)
    X = np.hstack([d["fp0"], d["fp1"]]).astype(np.float32)
    labels, groups, fam = d["sources"], d["groups"].astype(str), d["fam"]
    del d
    real, y = labels == "real", labels != "real"
    ri = np.flatnonzero(real)
    _, cal_pos = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(ri, groups=groups[ri]))
    cal = ri[cal_pos]
    rng = np.random.default_rng(0)
    for g in args.families:
        key = f"{args.kind} {g}"
        if key in results:
            continue
        gi = np.flatnonzero(fam == g)
        print(f"\n===== {g}: {len(gi)} songs, {len(set(groups[gi]))} groups =====", flush=True)
        folds = list(GroupKFold(3).split(gi, groups=groups[gi]))
        res = {f"{s:.2f}": [] for s in SHARES}
        flagged = {f"{s:.2f}": [] for s in SHARES}
        for k, (other, test) in enumerate(folds):
            other_groups = rng.permutation(sorted(set(groups[gi[other]])))
            for s in SHARES:
                keep = set(other_groups[: int(round(s * len(other_groups)))])
                use = ~np.isin(np.arange(len(y)), gi)
                use[gi[other][np.isin(groups[gi[other]], list(keep))]] = True
                use[cal] = False
                score = tt.train(args.kind, X, y, fam, use)
                cut = np.quantile(score(X[cal]), 0.95)
                res[f"{s:.2f}"].append(float(np.mean(score(X[gi[test]]) > cut)))
                flagged[f"{s:.2f}"].append(int(use[gi].sum()))
                print(f"  fold {k + 1}, share {s:.2f} ({int(use[gi].sum())} of its songs trained on): "
                      f"{100 * res[f'{s:.2f}'][-1]:.1f}% caught", flush=True)
        results[key] = {s: [float(np.mean(v)), float(np.mean(flagged[s]))] for s, v in res.items()}
        if not args.quick:
            out.write_text(json.dumps(results, indent=1))
    print("\nShare of a family's test fold caught, by how many of its songs were trained on (mean of 3 folds)")
    for key, r in results.items():
        vals = list(r.values())
        gain = 100 * (vals[-1][0] - vals[-2][0])
        print(f"  {key:16s}" + "".join(f"  {100 * v[0]:5.1f}% ({v[1]:5.0f})" for v in vals)
              + f"   last step {gain:+.1f} -> {'still rising' if gain >= 3 else 'levelled off'}")


if __name__ == "__main__":
    main()

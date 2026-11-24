r"""Combine the two logistic regressions: one boundary (better on new generators) and per family (fewer false alarms).

Both models are trained on the same songs in the same splits; then their scores are combined. Each score is first
turned into "share of real songs below it" (its percentile among held-out real scores), so the two scales match.
Rules (fixed before running):
  one boundary   LR, one yes/no boundary (as v2)
  per family     LR, one boundary per family
  max            the higher of the two percentiles ("either model says AI")
  mean           the mean of the two percentiles
  + comb         the same rules with the precise comb in both bands added to the max (the joint rule)
One cut-off per rule, set so 5% of real songs are flagged. Tests (as nonlinear_test.py):
  cv     5-fold cross-validation (real songs grouped by artist), every generator in training;
  logo   leave one generator family out; half of the held-out real fold sets the cut-offs, the other half
         measures false alarms;
  fresh  fresh44 (new real songs, Suno V5.5, MiniMax never seen, Lyria 3) + a real source kept out of training,
         with the final models and the cut-offs from cross-validation.
Songs: as nonlinear_test.py --extra-real features_hq\more_real --hold-real etree (live concerts kept out to test on).

  python -I combo_test.py                       # ~1.5 hours; reports\combo_test.json
  python -I combo_test.py --quick 3000 --tests cv
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

from nonlinear_test import load_all, pct, train  # noqa: E402

HERE = Path(__file__).resolve().parent
KINDS = ("lr", "family")
RULES = ["one boundary", "per family", "max", "mean", "one boundary + comb", "per family + comb", "max + comb"]


def combine(ref, p, s16, s44):
    """ref: real reference scores {"lr", "family", "s16", "s44"}; p: {"lr", "family"} scores -> {rule: score}."""
    a, b = pct(ref["lr"], p["lr"]), pct(ref["family"], p["family"])
    c = np.maximum(pct(ref["s16"], s16), pct(ref["s44"], s44))
    return {"one boundary": a, "per family": b, "max": np.maximum(a, b), "mean": (a + b) / 2,
            "one boundary + comb": np.maximum(a, c), "per family + comb": np.maximum(b, c),
            "max + comb": np.maximum.reduce([a, b, c])}


def show(title, table):
    print(f"\n{title}")
    cols = list(next(iter(table.values())))
    print(f"  {'':16s}" + "".join(f"{c[:13]:>14s}" for c in cols))
    for row, vals in table.items():
        print(f"  {row:16s}" + "".join(f"{100 * vals[c]:13.1f}%" for c in cols))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", nargs="+", default=["cv", "fresh", "logo"], choices=["cv", "fresh", "logo"])
    ap.add_argument("--quick", type=int, default=0)
    ap.add_argument("--fresh", default=str(HERE / "features_hq" / "unseen" / "fresh44.npz"))
    ap.add_argument("--extra-real", default=str(HERE / "features_hq" / "more_real"))
    ap.add_argument("--hold-real", nargs="*", default=["etree"])
    ap.add_argument("--out", default=str(HERE / "reports" / "combo_test.json"))
    args = ap.parse_args()

    X, labels, groups, s16, s44, fam, origin, held = load_all(args.quick, extra_real=args.extra_real,
                                                             hold=args.hold_real)
    real, y = labels == "real", labels != "real"
    out = {}

    if "cv" in args.tests or "fresh" in args.tests:
        t = time.time()
        oof = {k: np.zeros(len(y)) for k in KINDS}
        for i, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, labels, groups), 1):
            use = np.ones(len(y), bool)
            use[te] = False
            for k in KINDS:
                oof[k][te] = train(k, X, y, fam, use)(X[te])
            print(f"  cv fold {i} ({(time.time() - t) / 60:.1f} min)", flush=True)
        ref = {k: np.sort(oof[k][real]) for k in KINDS} | {"s16": np.sort(s16[real]), "s44": np.sort(s44[real])}
        sc = combine(ref, oof, s16, s44)
        cuts = {r: float(np.quantile(v[real], 0.95)) for r, v in sc.items()}
        rows = {g: labels == g for g in sorted(set(labels[y]))} | {"ALL AI": y}
        rows |= {f"real {o}": real & (origin == o) for o in sorted(set(origin[real]))} | {"ALL REAL": real}
        out["cv"] = {g: {r: float(np.mean(sc[r][m] > cuts[r])) for r in RULES} for g, m in rows.items()}
        show("Cross-validated (AI rows: caught; real rows: flagged)", out["cv"])

    if "fresh" in args.tests:
        models = {k: train(k, X, y, fam, np.ones(len(y), bool)) for k in KINDS}
        f = np.load(args.fresh)
        hi = f["src_sr"] >= 44100
        src = f["sources"].astype(str)
        Xf = np.hstack([f["fp0"], f["fp1"]])
        sf_ = combine(ref, {k: m(Xf) for k, m in models.items()}, f["sharp0"], f["sharp1"])
        rows = {s: (src == s) & hi for s in sorted(set(src))} | {"ALL REAL": np.char.startswith(src, "real") & hi}
        out["fresh"] = {s: {r: float(np.mean(sf_[r][m] > cuts[r])) for r in RULES} for s, m in rows.items()}
        for h, v in held.items():
            sh = combine(ref, {k: m(v["X"]) for k, m in models.items()}, v["s16"], v["s44"])
            out["fresh"][f"held {h}"] = {r: float(np.mean(sh[r] > cuts[r])) for r in RULES}
        show("fresh44 and the real source kept out of training (AI rows: caught; real rows: flagged)", out["fresh"])

    if "logo" in args.tests:
        t = time.time()
        fams = sorted(set(fam[y]))
        real_idx = np.flatnonzero(real)
        fold = np.empty(len(real_idx), int)
        for k, (_, te) in enumerate(GroupKFold(len(fams)).split(real_idx, groups=groups[real])):
            fold[te] = k
        rng = np.random.default_rng(0)
        out["logo"], fa, n = {}, [], []
        for i, g in enumerate(fams):
            cal = rng.permutation(real_idx[fold == i])
            set_cut, check = cal[: len(cal) // 2], cal[len(cal) // 2:]
            use = fam != g
            use[cal] = False
            models = {k: train(k, X, y, fam, use) for k in KINDS}
            gi = np.flatnonzero(fam == g)
            sc_of = lambda ii: combine(ref_l, {k: m(X[ii]) for k, m in models.items()}, s16[ii], s44[ii])  # noqa: E731
            ref_l = {k: np.sort(m(X[set_cut])) for k, m in models.items()} | {"s16": np.sort(s16[set_cut]),
                                                                               "s44": np.sort(s44[set_cut])}
            sset, sg, sc_ = sc_of(set_cut), sc_of(gi), sc_of(check)
            cl = {r: np.quantile(sset[r], 0.95) for r in RULES}
            out["logo"][g] = {r: float(np.mean(sg[r] > cl[r])) for r in RULES}
            fa.append({r: float(np.mean(sc_[r] > cl[r])) for r in RULES})
            n.append(len(gi))
            print(f"  left out {g:12s} " + " ".join(f"{100 * out['logo'][g][r]:5.1f}" for r in RULES)
                  + f"  ({(time.time() - t) / 60:.1f} min)", flush=True)
        n = np.array(n)
        out["logo"]["ALL AI"] = {r: float(n @ [out["logo"][g][r] for g in fams] / n.sum()) for r in RULES}
        out["logo"]["REAL flagged"] = {r: float(np.mean([a[r] for a in fa])) for r in RULES}
        show("Leave one family out (never trained on that family; last row: real songs flagged)", out["logo"])

    if not args.quick:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()

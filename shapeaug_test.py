r"""Can random EQ during training stop the shape feature from learning what FMA songs look like?

temporal_test.py found that adding the spectrum's overall shape (shape0 + shape1) catches more AI songs in
cross-validation, but flags 10-26% of new real songs (fresh44, live concerts) instead of 5%: the shape mostly tells
"mixed like an FMA song" from "not", not real from AI. Fix tried here: during training, each song's shape gets a
random smooth EQ curve, so mixing and recording style no longer separate the classes; only what an EQ cannot copy
(e.g. a generator's fixed artefacts) is left to learn. The curve is added in dB to the stored shape, so no download:
  tilt         -2..+2 dB per octave around 1 kHz
  low shelf    -6..+6 dB below 80-400 Hz
  high shelf   -8..+8 dB above 2-10 kHz
  low-pass     (variant "eq+lp" only, half the songs) -48 dB per octave above 11-20 kHz
then clipped at -100 dB (the spectrum's floor) and made relative to 5-8 kHz as before. 80% of the training songs
get a curve, 20% stay as they are; a new curve every time a model is trained. Test songs are never changed.
The fakeprint part is not changed: a smooth EQ barely moves it (it subtracts the spectrum's lower envelope).
Variants (the "no EQ" rows come from reports\temporal_test.json):
  eq       tilt + shelves
  eq+lp    tilt + shelves + random low-pass (also hides where a generator cuts off)
Songs, models, tests and cut-offs as temporal_test.py (avg+shape features). Saved after each model; a re-run skips
what is done.

  python -I shapeaug_test.py --quick 3000 --tests cv     # timing check
  python -I shapeaug_test.py                             # LR one boundary + per family; reports\shapeaug_test.json
  python -I shapeaug_test.py --strength 3 --variants eq  # stronger EQ (gains x3), saved as "eq x3"
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import temporal_test as tt  # noqa: E402
from nonlinear_test import NAMES  # noqa: E402

HERE = Path(__file__).resolve().parent
VARIANTS = ["eq", "eq+lp"]
F0 = np.arange(0, 8000, 100.0) + 50  # shape0 band centres (16 kHz audio)
F1 = np.arange(0, 22050, 100.0) + 50  # shape1 band centres (44.1 kHz audio)


def curves(rng, n, lowpass, strength=1.0):
    """n random EQ curves in dB, evaluated at F0 and F1 (the same curve for both bands)."""
    tilt = strength * rng.uniform(-2, 2, (n, 1))
    gl, fl = strength * rng.uniform(-6, 6, (n, 1)), np.exp(rng.uniform(np.log(80), np.log(400), (n, 1)))
    gh, fh = strength * rng.uniform(-8, 8, (n, 1)), np.exp(rng.uniform(np.log(2000), np.log(10000), (n, 1)))
    lp = rng.uniform(11000, 20000, (n, 1)) if lowpass else np.full((n, 1), np.inf)
    if lowpass:
        lp[rng.random(n) < 0.5] = np.inf
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731

    def at(f):
        lf = np.log2(f)
        c = tilt * (lf - np.log2(1000)) + gl * sig((np.log2(fl) - lf) / 0.5) + gh * sig((lf - np.log2(fh)) / 0.5)
        return c - 48 * np.maximum(0, lf - np.log2(lp))
    return at(F0), at(F1)


def rel_shape(s0, s1):
    return np.hstack([s - s[:, 50:80].mean(1, keepdims=True) for s in (s0, s1)])


def make_train(d, lowpass, strength=1.0, seed=0):
    """A train() for temporal_test.evaluate that gives the training songs' shape columns a random EQ."""
    base = tt.train
    sh = rel_shape(d["shape0"], d["shape1"])
    mu, sd = sh.mean(0), sh.std(0) + 1e-6  # the same scaling as temporal_test.matrices
    n_avg = d["fp0"].shape[1] + d["fp1"].shape[1]
    rng = np.random.default_rng(seed)

    def aug_train(kind, X, y, fam, use):
        idx = np.flatnonzero(use)
        Xa = X[idx].copy()
        c0, c1 = curves(rng, len(idx), lowpass, strength)
        s0 = np.maximum(d["shape0"][idx] + c0, -100.0)
        s1 = np.maximum(d["shape1"][idx] + c1, -100.0)
        new = ((rel_shape(s0, s1) - mu) / sd).astype(np.float32)
        sel = rng.random(len(idx)) < 0.8
        Xa[sel, n_avg:] = new[sel]
        return base(kind, Xa, y[idx], fam[idx], np.ones(len(idx), bool))
    return aug_train


def summary(results):
    base = json.loads((HERE / "reports" / "temporal_test.json").read_text())
    keys = [("cv", "ALL AI", "AI caught, cv"), ("logo", "ALL AI", "AI caught, family never seen"),
            ("logo", "REAL flagged", "real flagged, logo"), ("fresh", "ALL REAL", "fresh44 real flagged"),
            ("fresh", "held etree", "live concerts flagged"), ("fresh", "minimax3", "MiniMax"),
            ("fresh", "lyria3", "Lyria 3"), ("fresh", "suno_v5.5", "Suno V5.5")]
    names = sorted({v for r in results.values() for v in r}, key=lambda v: (len(v), v))
    cols = {"avg": ("base", "avg"), "shape, no EQ": ("base", "avg+shape")} | {f"shape, {v}": ("aug", v) for v in names}
    for kind in results:
        print(f"\n{NAMES[kind]}: classifier (joint)")
        print(f"  {'':30s}" + "".join(f"{c:>18s}" for c in cols))
        for test, row, label in keys:
            vals = []
            for src, s in cols.values():
                r = (base if src == "base" else results).get(kind, {}).get(s, {})
                vals.append(r.get(test, {}).get(row))
            print(f"  {label:30s}" + "".join(f"{100 * v[0]:9.1f}% ({100 * v[1]:4.0f}%)" if v else f"{'-':>18s}"
                                             for v in vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["lr", "family"], choices=list(NAMES))
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--tests", nargs="+", default=["cv", "fresh", "logo"], choices=["cv", "fresh", "logo"])
    ap.add_argument("--strength", type=float, default=1.0, help="multiply the EQ gains (tilt and shelves)")
    ap.add_argument("--quick", type=int, default=0)
    ap.add_argument("--features", default=str(HERE / "features_tm"))
    ap.add_argument("--out", default=str(HERE / "reports" / "shapeaug_test.json"))
    args = ap.parse_args()
    out = Path(args.out)
    results = json.loads(out.read_text()) if out.exists() and not args.quick else {}
    print("Loading ...")
    d, held, f = tt.load(Path(args.features), args.quick)
    X, Xh, Xf = tt.matrices(d, held, f)
    s = "avg+shape"
    base_train = tt.train
    for v0 in args.variants:
        v = v0 if args.strength == 1 else f"{v0} x{args.strength:g}"
        for kind in args.models:
            done = results.get(kind, {}).get(v, {})
            tests = [t for t in args.tests if t not in done]
            if not tests:
                continue
            print(f"\n===== {NAMES[kind]}, shape with random {v}: {', '.join(tests)} =====", flush=True)
            tt.train = base_train
            tt.train = make_train(d, lowpass=(v0 == "eq+lp"), strength=args.strength)
            r = tt.evaluate(kind, X[s], d, {h: x[s] for h, x in Xh.items()}, held, Xf[s], f, tests)
            tt.train = base_train
            results.setdefault(kind, {}).setdefault(v, {}).update(r)
            if not args.quick:
                out.write_text(json.dumps(results, indent=1))
    print("\n===== Summary =====")
    summary(results)


if __name__ == "__main__":
    main()

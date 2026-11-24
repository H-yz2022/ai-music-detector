r"""Step 2: can the classifier learn "evenly spaced peaks" instead of "peaks at these exact frequencies"?

The classifier learns where each generator's peaks sit. A generator it never saw, or a song played 1% faster, puts
the peaks elsewhere, so it is missed (never-seen families ~64%; speed +-1% breaks it, robust_test.py).
Idea: frequency-stretch augmentation. While training, half of the training songs (real and AI alike, chosen anew
for every model) have both fakeprints stretched along the frequency axis by a random factor s, uniform in
[1 - a, 1 + a]: fp_new(f) = fp(f / s), linear interpolation, 0 outside the band. A speed change by s moves the
peaks the same way, so the model sees the same pattern at many positions. No download: the stored fakeprints are
used (features_tm\, as autocorr_test.py), and no rows are added (memory).
Settings: a = 0 (as now), 0.03 (speed changes up to 3%), 0.10. Models: LR per family and LR one boundary.
Tests: as autocorr_test.py (cv, leave one family out, fresh44, live concerts, 1,077 new real songs), plus every
version of the robustness songs (features_hq\unseen\robust.npz: clean, speed -3..+3%, pitch, tempo, MP3, Ogg,
EQ, noise; never trained on), scored with the cv cut-off.
Rule fixed before running, for each model: a stretch setting is better than a = 0 if it catches at least 2 points
more of the never-seen families, flags at most 6% of the new real songs, and flags at most 1 point more than a = 0
on fresh44's real songs and the live concerts. Speed robustness is reported, not part of the rule.

  python -I stretch_test.py --quick 3000 --tests cv     # timing check
  python -I stretch_test.py                             # ~2 hours; reports\stretch_test.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

import temporal_test as tt  # noqa: E402
tt.SKIP |= {"tstd0", "tstd1", "tco0", "tco1", "shape0", "shape1", "stereo"}  # only the fakeprints are used
from nonlinear_test import NAMES  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = [("family", 0.0), ("family", 0.03), ("family", 0.10), ("lr", 0.0), ("lr", 0.03), ("lr", 0.10)]
KEEP = ("fp0", "fp1", "sharp0", "sharp1", "sources", "groups", "origin", "fam")


def band_freqs():
    b = joblib.load(HERE / "model_twoband.joblib")
    return [np.asarray(band["freqs"], dtype=np.float64) for band in b["bands"]]


def stretch_rows(x, freqs, s):
    """Each row of x (songs x bins) stretched along frequency by its own factor s: x_new(f) = x(f / s)."""
    df = freqs[1] - freqs[0]
    pos = (freqs[None, :] / s[:, None] - freqs[0]) / df  # where each new bin reads from, in old bins
    i0 = np.floor(pos).astype(np.int64)
    w = (pos - i0).astype(np.float32)
    n = x.shape[1]
    ok0, ok1 = (i0 >= 0) & (i0 < n), (i0 + 1 >= 0) & (i0 + 1 < n)
    rows = np.arange(len(x))[:, None]
    a = np.where(ok0, x[rows, np.clip(i0, 0, n - 1)], 0)
    b = np.where(ok1, x[rows, np.clip(i0 + 1, 0, n - 1)], 0)
    return ((1 - w) * a + w * b).astype(np.float32)


def make_train(amount, freqs, seed=0):
    base = tt.train
    n0 = len(freqs[0])
    rng = np.random.default_rng(seed)

    def aug_train(kind, X, y, fam, use):
        idx = np.flatnonzero(use)
        Xa = X[idx].copy()
        if amount > 0:
            sel = np.flatnonzero(rng.random(len(idx)) < 0.5)
            for start in range(0, len(sel), 250):  # in small chunks, to keep memory low
                c = sel[start:start + 250]
                s = rng.uniform(1 - amount, 1 + amount, len(c))
                Xa[c, :n0] = stretch_rows(Xa[c, :n0], freqs[0], s)
                Xa[c, n0:] = stretch_rows(Xa[c, n0:], freqs[1], s)
        return base(kind, Xa, y[idx], fam[idx], np.ones(len(idx), bool))
    return aug_train


def robust_sets():
    r = np.load(HERE / "features_hq" / "unseen" / "robust.npz")
    real = np.char.startswith(r["sources"].astype(str), "real")
    out = {}
    for c, cond in enumerate(r["conditions"].astype(str)):
        for who, m in (("real", real), ("ai", ~real)):
            out[f"robust {cond} | {who}"] = {"fp0": r["fp0"][c][m].astype(np.float32),
                                             "fp1": r["fp1"][c][m].astype(np.float32),
                                             "sharp0": r["sharp0"][c][m], "sharp1": r["sharp1"][c][m]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", nargs="+", default=["cv", "fresh", "logo"], choices=["cv", "fresh", "logo"])
    ap.add_argument("--quick", type=int, default=0)
    ap.add_argument("--features", default=str(HERE / "features_tm"))
    ap.add_argument("--out", default=str(HERE / "reports" / "stretch_test.json"))
    args = ap.parse_args()
    out = Path(args.out)
    results = json.loads(out.read_text()) if out.exists() and not args.quick else {}
    print("Loading ...", flush=True)
    d, held, f = tt.load(Path(args.features), args.quick)
    d = {k: d[k] for k in KEEP}
    held = {h: {k: v[k] for k in ("fp0", "fp1", "sharp0", "sharp1")} for h, v in held.items()}
    f = {k: f[k] for k in ("fp0", "fp1", "sharp0", "sharp1", "sources")}
    for p in sorted((Path(args.features) / "newreal").glob("*.npz")):
        x = np.load(p)
        ok = x["src_sr"] >= 44100
        held[p.stem] = {k: x[k][ok] for k in ("fp0", "fp1", "sharp0", "sharp1")}
    new = ["community", "jamendo", "mixtapes"]
    held |= robust_sets()

    def avg(x):
        return np.hstack([x["fp0"], x["fp1"]]).astype(np.float32)
    X, Xh, Xf = avg(d), {h: avg(v) for h, v in held.items()}, avg(f)
    freqs = band_freqs()
    assert len(freqs[0]) == d["fp0"].shape[1] and len(freqs[1]) == d["fp1"].shape[1]

    for kind, amount in RUNS:
        key = f"{kind} a={amount:g}"
        tests = [t for t in args.tests if t not in results.get(key, {})]
        if not tests:
            continue
        print(f"\n===== {NAMES[kind]}, stretch a={amount:g}: {', '.join(tests)} =====", flush=True)
        tt.train = make_train(amount, freqs)
        r = tt.evaluate(kind, X, d, Xh, held, Xf, f, tests)
        if "fresh" in r:
            n = np.array([r["fresh"][f"held {h}"][2] for h in new])
            r["fresh"]["ALL NEW REAL"] = [float(n @ [r["fresh"][f"held {h}"][j] for h in new] / n.sum())
                                          for j in (0, 1)] + [int(n.sum())]
        results.setdefault(key, {}).update(r)
        if not args.quick:
            out.write_text(json.dumps(results, indent=1))
    summary(results)


def summary(results):
    keys = [("cv", "ALL AI", "AI caught, cv"), ("logo", "ALL AI", "AI caught, family never seen"),
            ("logo", "REAL flagged", "real flagged, logo"), ("fresh", "ALL NEW REAL", "new real songs flagged"),
            ("fresh", "ALL REAL", "fresh44 real flagged"), ("fresh", "held etree", "live concerts flagged"),
            ("fresh", "minimax3", "MiniMax"), ("fresh", "lyria3", "Lyria 3"), ("fresh", "suno_v5.5", "Suno V5.5"),
            ("logo", "mureka", "Mureka, never seen"), ("logo", "lyria", "Lyria, never seen"),
            ("logo", "udio", "Udio, never seen"), ("logo", "sonauto", "Sonauto, never seen")]
    conds = ["clean", "speed -3%", "speed -1%", "speed +1%", "speed +3%", "pitch -1 st", "pitch +1 st", "tempo +5%",
             "MP3 128k", "EQ", "noise 30 dB"]
    keys += [("fresh", f"held robust {c} | ai", f"robust {c}: AI caught") for c in conds]
    keys += [("fresh", f"held robust {c} | real", f"robust {c}: real flagged") for c in ("clean", "speed +3%")]
    print("\nClassifier (with comb)\n  " + f"{'':34s}" + "".join(f"{k:>17s}" for k in results))
    for test, row, label in keys:
        vals = [results[k].get(test, {}).get(row) for k in results]
        if any(vals):
            print(f"  {label:34s}" + "".join(f"{100 * v[0]:8.1f}% ({100 * v[1]:4.0f}%)" if v else f"{'-':>17s}"
                                             for v in vals))
    for kind in ("family", "lr"):
        b = results.get(f"{kind} a=0", {})
        for amount in (0.03, 0.10):
            a = results.get(f"{kind} a={amount:g}", {})
            if all(t in a and t in b for t in ("logo", "fresh")):
                g = a["logo"]["ALL AI"][0] - b["logo"]["ALL AI"][0]
                nr = a["fresh"]["ALL NEW REAL"][0]
                df = max(a["fresh"][r][0] - b["fresh"][r][0] for r in ("ALL REAL", "held etree"))
                ok = g >= 0.02 and nr <= 0.06 and df <= 0.01
                print(f"Rule ({NAMES[kind]}, a={amount:g} vs a=0): never-seen families {100 * g:+.1f} points, new "
                      f"real songs flagged {100 * nr:.1f}%, fresh44 / live concerts up to {100 * df:+.1f} points -> "
                      f"{'BETTER' if ok else 'NOT BETTER'}")


if __name__ == "__main__":
    main()

r"""A fakeprint that does not care where the peaks sit: its autocorrelation.

The fakeprint records where each peak of the average spectrum is. A generator never seen puts its peaks elsewhere,
which is one reason only ~65% of never-seen families are caught. The autocorrelation of the fakeprint describes the
pattern instead ("peaks every 86 Hz"), whatever the exact positions. Built from the stored fakeprints (features_tm\),
no download:
  ac   autocorrelation of each band's fakeprint (mean removed, divided by its value at lag 0), lags up to 1000 Hz:
       1,024 values for the 1-8 kHz band (0.98 Hz steps), 372 for the 5-16 kHz band (2.7 Hz steps);
       standardized per value over the training songs (no labels used)
Feature sets: avg (the fakeprints, as now), ac (autocorrelation only), avg+ac.
Models: LR per family and LR one boundary on all three; LightGBM on ac alone (small enough to be quick).
Songs, tests and cut-offs as temporal_test.py (cv, leave one family out, fresh44 and live concerts), plus the
1,077 never-used real songs of newreal_test.py (features_tm\newreal\).
Rule fixed before running: avg+ac is better than avg (LR per family) if it catches at least 2 points more of the
never-seen families, flags at most 6% of the new real songs, and flags at most 1 point more than avg on fresh44's
real songs and the live concerts.

  python -I autocorr_test.py --quick 3000 --tests cv     # timing check
  python -I autocorr_test.py                             # ~1.5 hours; reports\autocorr_test.json
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
LAGS = {"fp0": 1024, "fp1": 372}  # lags up to ~1000 Hz (0.98 and 2.69 Hz per bin)
RUNS = [("family", "avg"), ("family", "ac"), ("family", "avg+ac"), ("lr", "avg"), ("lr", "ac"), ("lr", "avg+ac"),
        ("lgbm", "ac")]


def autocorr(fp, lags):
    x = fp.astype(np.float64) - fp.mean(1, keepdims=True)
    n = 1 << int(np.ceil(np.log2(2 * x.shape[1])))
    p = np.fft.rfft(x, n)
    r = np.fft.irfft(p * np.conj(p), n)[:, : lags + 1]
    return (r[:, 1:] / (r[:, :1] + 1e-12)).astype(np.float32)


def add_ac(x):
    x["ac"] = np.hstack([autocorr(x[k], n) for k, n in LAGS.items()])


def matrices(d, held, f):
    for x in [d, f, *held.values()]:
        add_ac(x)
    mu, sd = d["ac"].mean(0), d["ac"].std(0) + 1e-6

    def build(x):
        avg = np.hstack([x["fp0"], x["fp1"]]).astype(np.float32)
        ac = ((x["ac"] - mu) / sd).astype(np.float32)
        return {"avg": avg, "ac": ac, "avg+ac": np.hstack([avg, ac])}
    return build(d), {h: build(v) for h, v in held.items()}, build(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", nargs="+", default=["cv", "fresh", "logo"], choices=["cv", "fresh", "logo"])
    ap.add_argument("--quick", type=int, default=0)
    ap.add_argument("--features", default=str(HERE / "features_tm"))
    ap.add_argument("--out", default=str(HERE / "reports" / "autocorr_test.json"))
    args = ap.parse_args()
    out = Path(args.out)
    results = json.loads(out.read_text()) if out.exists() and not args.quick else {}
    print("Loading ...")
    d, held, f = tt.load(Path(args.features), args.quick)
    for p in sorted((Path(args.features) / "newreal").glob("*.npz")):
        x = dict(np.load(p))
        ok = x["src_sr"] >= 44100
        held[p.stem] = {k: v[ok] for k, v in x.items() if getattr(v, "ndim", 0) and len(v) == len(ok)}
    new = [h for h in held if h != "etree"]
    X, Xh, Xf = matrices(d, held, f)
    for kind, s in RUNS:
        key = f"{kind} {s}"
        tests = [t for t in args.tests if t not in results.get(key, {})]
        if not tests:
            continue
        print(f"\n===== {NAMES[kind]}, {s} ({X[s].shape[1]} features): {', '.join(tests)} =====", flush=True)
        r = tt.evaluate(kind, X[s], d, {h: v[s] for h, v in Xh.items()}, held, Xf[s], f, tests)
        if "fresh" in r:
            n = np.array([r["fresh"][f"held {h}"][2] for h in new])
            r["fresh"]["ALL NEW REAL"] = [float(n @ [r["fresh"][f"held {h}"][j] for h in new] / n.sum())
                                          for j in (0, 1)] + [int(n.sum())]
        results.setdefault(key, {}).update(r)
        if not args.quick:
            out.write_text(json.dumps(results, indent=1))

    keys = [("cv", "ALL AI", "AI caught, cv"), ("logo", "ALL AI", "AI caught, family never seen"),
            ("logo", "REAL flagged", "real flagged, logo"), ("fresh", "ALL NEW REAL", "new real songs flagged"),
            ("fresh", "ALL REAL", "fresh44 real flagged"), ("fresh", "held etree", "live concerts flagged"),
            ("fresh", "minimax3", "MiniMax"), ("fresh", "lyria3", "Lyria 3"), ("fresh", "suno_v5.5", "Suno V5.5"),
            ("logo", "mureka", "Mureka, never seen"), ("logo", "lyria", "Lyria, never seen"),
            ("logo", "udio", "Udio, never seen"), ("logo", "mubert", "Mubert, never seen")]
    print("\nClassifier (with comb)\n  " + f"{'':30s}" + "".join(f"{k:>17s}" for k in results))
    for test, row, label in keys:
        vals = [results[k].get(test, {}).get(row) for k in results]
        if any(vals):
            print(f"  {label:30s}" + "".join(f"{100 * v[0]:8.1f}% ({100 * v[1]:4.0f}%)" if v else f"{'-':>17s}"
                                             for v in vals))
    a, b = results.get("family avg+ac", {}), results.get("family avg", {})
    if all(t in a and t in b for t in ("logo", "fresh")):
        g = a["logo"]["ALL AI"][0] - b["logo"]["ALL AI"][0]
        nr = a["fresh"]["ALL NEW REAL"][0]
        df = max(a["fresh"][r][0] - b["fresh"][r][0] for r in ("ALL REAL", "held etree"))
        ok = g >= 0.02 and nr <= 0.06 and df <= 0.01
        print(f"\nRule (LR per family, avg+ac vs avg): never-seen families {100 * g:+.1f} points, new real songs "
              f"flagged {100 * nr:.1f}%, fresh44 / live concerts up to {100 * df:+.1f} points -> "
              f"{'BETTER' if ok else 'NOT BETTER'}")


if __name__ == "__main__":
    main()

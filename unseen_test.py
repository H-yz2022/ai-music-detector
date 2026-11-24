r"""Unseen generators: does a detector trained only on FMA (real) and SONICS
(Suno v2-v3.5, Udio) flag songs from generators it never saw?

Test songs come from the Echoes dataset (features\unseen\echoes.npz, made by
`build_fakeprints.py echoes`): 12 generators, including diffusion-based ones
and newer Suno. Same format as everything else (16 kHz, 40 kbps MP3, first 30 s,
1-8 kHz band).

Three detectors, each with a cut-off that flags 5% of the real training songs:
- classifier: the paper's logistic regression; cut-off chosen by
  cross-validation on the training pool (as in cutoff_test.py);
- comb: the training-free comb score with bump removal;
- precise comb: fakeprint.comb_sharpness near the comb score's rough spacing,
  i.e. does the comb repeat at one *exact* spacing (first version);
- top 5: fakeprint.precise_comb, the same at the 5 strongest repeat distances
  (the version detect.py uses). Settings were fixed before each run.
"either" = classifier or top 5. SONICS rows use cross-validated P(AI),
so the classifier never scores a song it was trained on.

  python -I unseen_test.py
"""
import argparse
import collections
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import fakeprint as fpm  # noqa: E402
from cutoff_test import cross_validated, default_groups  # noqa: E402
from detect import make_model  # noqa: E402
from scale_test import load_features, load_test  # noqa: E402

HERE = Path(__file__).resolve().parent


def comb_rows(freqs, fps):
    """(comb score, spacing, sharpness, exact spacing, top-5 sharpness, top-5 exact spacing) per fakeprint."""
    out = []
    for fp in fps.astype(np.float64):
        c, s = fpm.comb_score(freqs, fp, bump_hz=fpm.BUMP_HZ)
        out.append((c, s, *fpm.comb_sharpness(freqs, fp, s), *fpm.precise_comb(freqs, fp)))
    return out


def _comb_part(job):  # runs in a worker process; loads its own slice to keep RAM low
    path, start, stop = job
    d = np.load(path)
    return comb_rows(d["freqs"], d["fp"][start:stop])


def comb_file(path, workers, chunk=1000):
    n = len(np.load(path)["gens"])
    jobs = [(str(path), i, min(i + chunk, n)) for i in range(0, n, chunk)]
    with ProcessPoolExecutor(workers) as pool:
        return np.array([r for part in pool.map(_comb_part, jobs) for r in part])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=str(HERE / "features"))
    ap.add_argument("--unseen", default=str(HERE / "features" / "unseen" / "echoes.npz"))
    ap.add_argument("--test", default=str(HERE / "data_fresh"), help="fresh real songs, for the false-alarm rate")
    ap.add_argument("--false-alarms", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--fmin", type=float, default=1000.0)
    ap.add_argument("--fmax", type=float, default=8000.0)
    ap.add_argument("--max-duration", type=float, default=30.0)
    args = ap.parse_args()

    print("Training pool:")
    freqs, X, gens, names = load_features(args.features, args)
    u = np.load(args.unseen)
    if not np.array_equal(freqs, u["freqs"]):
        sys.exit("Unseen-generator fakeprints have different frequency bins from the training pool.")
    Xu, ugens = u["fp"].astype(np.float32), u["gens"]
    print(f"Unseen generators: {len(ugens)} songs from {len(set(ugens))} generators")
    print("Fresh real songs (data_fresh):")
    _, Xt, tgens, _ = load_test(Path(args.test), args)
    Xr = Xt[tgens == "real"]

    print("\nCross-validation for the classifier cut-off ...")
    oof = cross_validated(X, gens, default_groups(names, gens))
    model = make_model().fit(X, gens != "real")
    pu, pr = model.predict_proba(Xu)[:, 1], model.predict_proba(Xr)[:, 1]
    del X

    print("Comb scores and sharpness (a few minutes) ...")
    cp = np.concatenate([comb_file(p, args.workers) for p in sorted(Path(args.features).glob("*.npz"))])
    cu, cr = comb_file(args.unseen, args.workers), np.array(comb_rows(freqs, Xr))

    q = 1 - args.false_alarms / 100
    real = gens == "real"
    cut, comb_cut, sharp_cut, top_cut = (float(np.quantile(v, q))
                                         for v in (oof[real], cp[real, 0], cp[real, 2], cp[real, 4]))
    print(f"\nCut-offs (each flags {args.false_alarms:g}% of real training songs): classifier P(AI) > {cut:.3f}, "
          f"comb > {comb_cut:.3f}, precise comb > {sharp_cut:.2f}, precise comb (top 5) > {top_cut:.2f}")

    rows = [("REAL (fresh)", pr, cr), ("SONICS suno*", oof[gens == "suno"], cp[gens == "suno"]),
            ("SONICS udio*", oof[gens == "udio"], cp[gens == "udio"])]
    rows += [(g, pu[ugens == g], cu[ugens == g]) for g in sorted(set(ugens))]
    rows += [("ALL ECHOES", pu, cu)]
    print(f"\n{'generator':14s}{'songs':>6s}{'classifier':>12s}{'comb':>7s}{'precise':>9s}{'top 5':>7s}"
          f"{'either':>8s}{'exact spacing (songs)':>24s}")
    for g, p, c in rows:
        precise, top5 = c[:, 2] > sharp_cut, c[:, 4] > top_cut
        spacing = "-"
        if top5.sum() >= 3 and g != "ALL ECHOES":
            top = collections.Counter(np.round(c[top5, 5], 2)).most_common(1)[0]
            spacing = f"{top[0]:.2f} Hz ({top[1]})"
        print(f"{g:14s}{len(p):6d}{100 * np.mean(p > cut):11.0f}%{100 * np.mean(c[:, 0] > comb_cut):6.0f}%"
              f"{100 * np.mean(precise):8.0f}%{100 * np.mean(top5):6.0f}%{100 * np.mean((p > cut) | top5):7.0f}%"
              f"{spacing:>24s}")
    print("(precise = precise comb near the rough spacing; top 5 = precise comb at the 5 strongest repeat distances;"
          " either = classifier or top 5.\n REAL: 'flagged' is the false-alarm rate. *SONICS songs are in the "
          "training pool: classifier uses cross-validated P(AI).)")


if __name__ == "__main__":
    main()

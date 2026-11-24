r"""Idea 3, the test: does information the fakeprint throws away (temporal_build.py) help, for each of six models?

Feature sets, from the same songs and the same pass (features_tm\):
  avg          the two fakeprints (fp0 + fp1, 11,254 values): what every model used so far
  avg+shape    + the spectrum's overall shape (shape0 + shape1: mean dB per 100 Hz, each band relative to its
               5-8 kHz level; 301 values)
  avg+stereo   + stereo width (side minus mid per 100 Hz, 0-16 kHz, from the original file; 160 values)
  avg+time     + the time features (tstd0 + tstd1: 11,254 values; tco0 + tco1: 602 values)
  all          all of them
The new features are put on the fakeprint's scale before training: tstd divided by its overall standard
deviation; tco, shape and stereo standardized per value (mean and standard deviation over the training songs, no
labels used). The real training songs are mostly FMA small, so a feature that only learns their look shows up as
false alarms on fresh44's real songs and on the live concerts, never trained on.
Models: the six of nonlinear_test.py (LR one boundary, LR per family, LightGBM, neural net, random forest, RBF SVM),
same settings. Songs: as nonlinear_test.py --extra-real features_hq\more_real --hold-real etree: FMA small +
Echoes (the train_twoband.py songs), more44, more_real at 44.1 kHz or more with the live concerts (etree) kept out.
Tests, all at 5% false alarms on real songs (as nonlinear_test.py):
  cv     5-fold cross-validation (real songs grouped by artist)
  logo   leave one generator family out (half the held-out real fold sets the cut-off, half measures false alarms)
  fresh  fresh44 (new real songs, Suno V5.5, MiniMax, Lyria 3) and the live concerts, cut-off from cv
"joint" adds the precise comb in both bands (one cut-off on the combined score). Settings fixed before running.
Results are saved after each model, and a re-run skips what is done.

  python -I temporal_test.py --quick 3000 --models lr --tests cv     # timing check
  python -I temporal_test.py                                         # everything; reports\temporal_test.json
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
from nonlinear_test import NAMES, pct, train  # noqa: E402

HERE = Path(__file__).resolve().parent
SETS = ["avg", "avg+shape", "avg+stereo", "avg+time", "all"]


SKIP = set()  # feature keys not to load, to save memory (e.g. tests that only use the fakeprints)


def npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files if k not in SKIP}


def read(paths):
    parts = [npz(p) for p in paths]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if all(k in p for p in parts)}


def load(tm, quick, hold=("etree",)):
    """Training songs and test songs from features_tm\\. Returns (train dict, held dict, fresh dict)."""
    from cutoff_test import artists_of
    from more44_test import FAMILY
    rows = []
    allowed = {}
    for which, lq, hq in (("fma_small", "features/fma_small.npz", "features_hq/fma_small.npz"),
                          ("echoes", "features/unseen/echoes.npz", "features_hq/echoes.npz")):
        allowed[which] = set(np.load(HERE / lq)["names"].astype(str)) & set(np.load(HERE / hq)["names"].astype(str))
    for which in ("fma_small", "echoes"):
        d = read(sorted((tm / which).glob("part_*.npz")))
        keep = np.isin(d["names"].astype(str), list(allowed[which]))
        d = {k: v[keep] for k, v in d.items()}
        names, labels = d["names"].astype(str), d["sources"].astype(str)
        art = artists_of(names, labels, str(HERE / "features" / "fma_artists.csv"))
        d["groups"] = np.array([f"artist{a}" if a is not None else n for a, n in zip(art, names)], dtype=object)
        d["origin"] = np.full(len(names), "fma_small" if which == "fma_small" else "ai", dtype=object)
        rows.append(d)
    new = read(sorted((tm / "more44").glob("*.npz")))
    new["origin"] = np.full(len(new["names"]), "ai", dtype=object)
    rows.append(new)
    for p in sorted((tm / "more_real").glob("*.npz")):
        d = npz(p)
        ok = d["src_sr"] >= 44100
        rows.append({k: v[ok] for k, v in d.items() if getattr(v, "ndim", 0) and len(v) == len(ok)})
    keys = [k for k in ["fp0", "fp1", "tstd0", "tstd1", "tco0", "tco1", "shape0", "shape1", "stereo", "sharp0",
                        "sharp1", "sources", "groups", "origin"] if k not in SKIP]
    d = {k: np.concatenate([r[k] for r in rows]) for k in keys}
    d["sources"], d["origin"] = d["sources"].astype(str), d["origin"].astype(str)
    held = {}
    for h in hold:
        m = d["origin"] == h
        held[h] = {k: v[m] for k, v in d.items()}
        d = {k: v[~m] for k, v in d.items()}
    if quick:
        keep = np.sort(np.random.default_rng(0).choice(len(d["sources"]), quick, replace=False))
        d = {k: v[keep] for k, v in d.items()}
    d["fam"] = np.array([FAMILY.get(s, s) for s in d["sources"]])
    f = npz(tm / "fresh44.npz")
    hi = f["src_sr"] >= 44100
    f = {k: v[hi] for k, v in f.items() if getattr(v, "ndim", 0) and len(v) == len(hi)}
    real = d["sources"] == "real"
    print(f"  {len(real)} training songs: {int(real.sum())} real ("
          + ", ".join(f"{o} {int((d['origin'] == o).sum())}" for o in sorted(set(d['origin'])) if o != "ai")
          + f"), {int((~real).sum())} AI from {len(set(d['fam'][~real]))} families; held out: "
          + ", ".join(f"{h} {len(v['sources'])}" for h, v in held.items()) + f"; fresh44 {len(f['sources'])}")
    return d, held, f


def matrices(d, held, f):
    """Feature matrices per set, with the time features scaled from the training songs."""
    s_std = float(np.concatenate([d["tstd0"].ravel(), d["tstd1"].ravel()]).std())
    tco = np.hstack([d["tco0"], d["tco1"]])
    mu, sd = tco.mean(0), tco.std(0) + 1e-6
    print(f"  scaling: tstd / {s_std:.3f} dB; tco, shape and stereo standardized per value")

    def rel_shape(x):  # each band's shape relative to its own 5-8 kHz level (so loudness drops out)
        return np.hstack([s - s[:, 50:80].mean(1, keepdims=True) for s in (x["shape0"], x["shape1"])])
    sh = rel_shape(d)
    mu_sh, sd_sh = sh.mean(0), sh.std(0) + 1e-6
    mu_st, sd_st = d["stereo"].mean(0), d["stereo"].std(0) + 1e-6

    def build(x):
        avg = np.hstack([x["fp0"], x["fp1"]]).astype(np.float32)
        tm = np.hstack([x["tstd0"] / s_std, x["tstd1"] / s_std, (np.hstack([x["tco0"], x["tco1"]]) - mu) / sd])
        shp = ((rel_shape(x) - mu_sh) / sd_sh).astype(np.float32)
        st = ((x["stereo"] - mu_st) / sd_st).astype(np.float32)
        return {"avg": avg, "avg+shape": np.hstack([avg, shp]), "avg+stereo": np.hstack([avg, st]),
                "avg+time": np.hstack([avg, tm.astype(np.float32)]),
                "all": np.hstack([avg, tm.astype(np.float32), shp, st])}
    return build(d), {h: build(v) for h, v in held.items()}, build(f)


def evaluate(kind, X, d, Xh, held, Xf, f, tests):
    labels, groups, fam, origin = d["sources"], d["groups"], d["fam"], d["origin"]
    s16, s44 = d["sharp0"], d["sharp1"]
    real, y = labels == "real", labels != "real"
    res, t = {}, time.time()
    if "cv" in tests or "fresh" in tests:
        oof = np.zeros(len(y))
        for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, labels, groups), 1):
            use = np.ones(len(y), bool)
            use[te] = False
            oof[te] = train(kind, X, y, fam, use)(X[te])
            print(f"    cv fold {k} ({(time.time() - t) / 60:.1f} min)", flush=True)
        cut = float(np.quantile(oof[real], 0.95))
        refs = {"clf": np.sort(oof[real]), "s16": np.sort(s16[real]), "s44": np.sort(s44[real])}
        joint = joint_score(refs, oof, s16, s44)
        jcut = float(np.quantile(joint[real], 0.95))
        rows = {g: labels == g for g in sorted(set(labels[y]))} | {"ALL AI": y}
        rows |= {f"real {o}": real & (origin == o) for o in sorted(set(origin[real]))}
        res["cv"] = {g: [float(np.mean(oof[m] > cut)), float(np.mean(joint[m] > jcut))] for g, m in rows.items()}
    if "fresh" in tests:
        score = train(kind, X, y, fam, np.ones(len(y), bool))
        src = f["sources"].astype(str)
        p = score(Xf)
        flag, jflag = p > cut, joint_score(refs, p, f["sharp0"], f["sharp1"]) > jcut
        rows = {s: src == s for s in sorted(set(src))} | {"ALL REAL": np.char.startswith(src, "real")}
        res["fresh"] = {s: [float(flag[m].mean()), float(jflag[m].mean()), int(m.sum())] for s, m in rows.items()}
        for h, v in held.items():
            ph = score(Xh[h])
            res["fresh"][f"held {h}"] = [float(np.mean(ph > cut)),
                                         float(np.mean(joint_score(refs, ph, v["sharp0"], v["sharp1"]) > jcut)),
                                         int(len(ph))]
        print(f"    fresh done ({(time.time() - t) / 60:.1f} min)", flush=True)
    if "logo" in tests:
        fams = sorted(set(fam[y]))
        real_idx = np.flatnonzero(real)
        fold = np.empty(len(real_idx), int)
        for k, (_, te) in enumerate(GroupKFold(len(fams)).split(real_idx, groups=groups[real])):
            fold[te] = k
        rng = np.random.default_rng(0)
        res["logo"], fa = {}, []
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
            print(f"    left out {g:12s} {100 * res['logo'][g][0]:5.1f}%  ({(time.time() - t) / 60:.1f} min)", flush=True)
        n = np.array([v[2] for v in res["logo"].values()])
        res["logo"]["ALL AI"] = [float(n @ [v[0] for v in res["logo"].values()] / n.sum()),
                                 float(n @ [v[1] for v in res["logo"].values()] / n.sum()), int(n.sum())]
        res["logo"]["REAL flagged"] = [float(np.mean([a[0] for a in fa])), float(np.mean([a[1] for a in fa])),
                                       int(len(real_idx))]
    return res


def summary(results):
    keys = [("cv", "ALL AI", "AI caught, cv"), ("logo", "ALL AI", "AI caught, family never seen"),
            ("logo", "REAL flagged", "real flagged, logo"), ("fresh", "ALL REAL", "fresh44 real flagged"),
            ("fresh", "held etree", "live concerts flagged"), ("fresh", "minimax3", "MiniMax"),
            ("fresh", "lyria3", "Lyria 3"), ("cv", "mureka_v9", "Mureka, cv"), ("cv", "lyria_pro3", "Lyria Pro 3, cv"),
            ("cv", "mubert", "Mubert, cv"), ("logo", "mureka", "Mureka, never seen"), ("logo", "lyria", "Lyria, never seen")]
    for kind, by_set in results.items():
        print(f"\n{NAMES[kind]}: classifier (joint)")
        print(f"  {'':30s}" + "".join(f"{s:>18s}" for s in SETS))
        for test, row, label in keys:
            vals = [by_set.get(s, {}).get(test, {}).get(row) for s in SETS]
            if any(v is not None for v in vals):
                print(f"  {label:30s}" + "".join(f"{100 * v[0]:9.1f}% ({100 * v[1]:4.0f}%)" if v else f"{'-':>18s}"
                                                 for v in vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["lr", "family", "lgbm", "mlp", "rf", "svm"], choices=list(NAMES))
    ap.add_argument("--sets", nargs="+", default=SETS, choices=SETS)
    ap.add_argument("--tests", nargs="+", default=["cv", "fresh", "logo"], choices=["cv", "fresh", "logo"])
    ap.add_argument("--quick", type=int, default=0)
    ap.add_argument("--features", default=str(HERE / "features_tm"))
    ap.add_argument("--out", default=str(HERE / "reports" / "temporal_test.json"))
    args = ap.parse_args()
    out = Path(args.out)
    results = json.loads(out.read_text()) if out.exists() and not args.quick else {}
    print("Loading ...")
    d, held, f = load(Path(args.features), args.quick)
    X, Xh, Xf = matrices(d, held, f)
    for s in args.sets:  # all models on one feature set, then the next set
        for kind in args.models:
            done = results.get(kind, {}).get(s, {})
            tests = [t for t in args.tests if t not in done]
            if not tests:
                continue
            print(f"\n===== {NAMES[kind]}, {s} ({X[s].shape[1]} features): {', '.join(tests)} =====", flush=True)
            r = evaluate(kind, X[s], d, {h: v[s] for h, v in Xh.items()}, held, Xf[s], f, tests)
            results.setdefault(kind, {}).setdefault(s, {}).update(r)
            if not args.quick:
                out.write_text(json.dumps(results, indent=1))
    print("\n===== Summary =====")
    summary(results)


if __name__ == "__main__":
    main()

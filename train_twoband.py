r"""Train the two-band classifier for `detect.py predict` (44.1 kHz input).

Each song gives two fakeprints, side by side (7,167 + 4,087 features):
- 1-8 kHz band: 16 kHz audio after the 40 kbps MP3 conversion (features\);
- 5-16 kHz band: 44.1 kHz audio after the 192 kbps MP3 conversion (features_hq\).
Training songs: the 12,357 songs that exist in both (7,894 real FMA clips, 4,463
Echoes songs from 12 generators). SONICS is left out: it only exists at 16 kHz.
twoband_test.py measured this setup: 96% cross-validated, 86% on a never-seen
generator, at 5% false alarms.

The cut-off comes from 5-fold cross-validation (real clips grouped by artist):
it flags 5% of real songs. The precise-comb cut-offs for both bands (5% of real
songs each) are saved too, so predict can say when a comb stands out.

It also saves a spacing bank (known_spacing_test.py): the exact comb spacing of every generator with a clear
comb (training generators plus --bank-extra songs), with real-song scores, for predict --bank (bank_check.py).

  python -I train_twoband.py --extra-dir features_hq\more_real
      # predict's default, model_twoband.joblib: v1 plus the varied real songs from more_real.py (44.1 kHz or more only)
  python -I train_twoband.py --out model_twoband_v1_old.joblib     # the first v1 (FMA small real songs only)
  python -I train_twoband.py --extra-dir features_hq\more44 --out model_twoband_v2.joblib
      # also the songs from more44_test.py (Udio, Mureka, HeartMuLa, Sonauto, Lyria Pro 3)
  python -I train_twoband.py --extra-dir features_hq\more44 --family --out model_twoband_family.joblib
      # the same, with one boundary per generator family
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

from cutoff_test import artists_of  # noqa: E402
from detect import ai_proba, joint_score  # noqa: E402
from hq_test import content_top, load, precise_all  # noqa: E402
from logo_test import fit  # noqa: E402
from fakeprint import known_comb, precise_comb  # noqa: E402

HERE = Path(__file__).resolve().parent
BANK_MIN = 8.0  # a generator joins the spacing bank if its mean fakeprint's precise comb is at least this (real: ~4.4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hq", nargs="+", default=[str(HERE / "features_hq" / "fma_small.npz"),
                                                str(HERE / "features_hq" / "echoes.npz")])
    ap.add_argument("--lq", nargs="+", default=[str(HERE / "features" / "fma_small.npz"),
                                                str(HERE / "features" / "unseen" / "echoes.npz")])
    ap.add_argument("--artists", default=str(HERE / "features" / "fma_artists.csv"))
    ap.add_argument("--cache", default=str(HERE / "features_hq" / "precise_cache.npz"))
    ap.add_argument("--false-alarms", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=str(HERE / "model_twoband.joblib"))
    ap.add_argument("--extra-dir", nargs="+", default=None,
                    help="also train on the songs in these folders or .npz files (features_hq\\more44 from "
                         "more44_test.py: AI songs; features_hq\\more_real from more_real.py: real songs)")
    ap.add_argument("--family", action="store_true", help="one boundary per generator family (multi-class)")
    ap.add_argument("--bank-extra", nargs="+", default=[str(HERE / "features_hq" / "more44")],
                    help="AI songs used only to learn comb spacings for the spacing bank (not to train the classifier)")
    args = ap.parse_args()

    print("Loading both bands ...")
    fh, Xh, gh, nh, coarse, edges = load(args.hq, "spec")
    fl, Xl, gl, nl = load(args.lq, "fp")
    common = sorted(set(nh) & set(nl))
    ih, il = {n: i for i, n in enumerate(nh)}, {n: i for i, n in enumerate(nl)}
    sel_h = [ih[n] for n in common]
    Xh, Xl, labels, coarse = Xh[sel_h], Xl[[il[n] for n in common]], gh[sel_h], coarse[sel_h]
    names = np.array(common)
    real = labels == "real"
    y = ~real
    print(f"  {len(names)} songs: {int(real.sum())} real, {int(y.sum())} AI from {len(set(labels[y]))} generators")

    # Settings of each band, read from the feature files (predict must make the same fakeprints)
    bands = []
    for path, f in ((args.lq[0], fl), (args.hq[0], fh)):
        d = np.load(path)
        sr = int(d["sr"])
        bands.append({"sr": sr, "fmin": float(d["fmin"]), "fmax": float(d["fmax"]),
                      "mp3_level": float(d["mp3_level"]) if "mp3_level" in d.files else 0.8, "freqs": f})
        if "match_sr" in d.files and int(d["match_sr"]) != sr:
            sys.exit(f"{path}: MP3 rate {int(d['match_sr'])} differs from analysis rate {sr}")
    max_duration = float(np.load(args.hq[0])["max_duration"])

    cache = Path(args.cache)
    c = np.load(cache) if cache.exists() else None
    if c is not None and np.array_equal(c["names"], names):
        s16, s44 = c["s16"], c["s44"]
    else:
        print("Precise comb in both bands ...")
        s16, s44 = precise_all(fl, Xl, args.workers), precise_all(fh, Xh, args.workers)
    q = 1 - args.false_alarms / 100

    X = np.hstack([Xl, Xh])
    del Xl, Xh
    art = artists_of(names, labels, args.artists)
    groups = np.array([f"artist{a}" if a is not None else n for a, n in zip(art, names)], dtype=object)
    top = content_top(coarse, edges)
    origin = np.where(real, "fma_small", "echoes").astype(object)
    if args.extra_dir:  # more songs (more44_test.py: AI, more_real.py: real), fakeprints and combs already made
        files = [f for d in args.extra_dir for f in (sorted(Path(d).glob("*.npz")) if Path(d).is_dir() else [Path(d)])]
        parts = []
        for f in files:
            d = dict(np.load(f))
            keep = d["src_sr"] >= bands[1]["sr"] if "src_sr" in d else np.ones(len(d["names"]), bool)
            if not keep.all():  # predict sends files below 44.1 kHz to the 16 kHz model, so leave them out
                print(f"  {f.name}: left out {int((~keep).sum())} songs below {bands[1]['sr']} Hz")
            d = {k: v[keep] if v.ndim and len(v) == len(keep) else v for k, v in d.items()}
            if "origin" not in d:
                d["origin"] = np.full(int(keep.sum()), f.stem, dtype=object)
            parts.append(d)
        X = np.vstack([X] + [np.hstack([p["fp0"], p["fp1"]]).astype(np.float32) for p in parts])
        labels = np.concatenate([labels] + [p["sources"] for p in parts])
        groups = np.concatenate([groups] + [p["groups"].astype(object) for p in parts])
        origin = np.concatenate([origin] + [p["origin"].astype(object) for p in parts])
        s16 = np.concatenate([s16] + [p["sharp0"] for p in parts])
        s44 = np.concatenate([s44] + [p["sharp1"] for p in parts])
        top = np.concatenate([top] + [p["top"] for p in parts])
        real, y = labels == "real", labels != "real"
        print(f"  + {sum(len(p['sources']) for p in parts)} songs from {', '.join(args.extra_dir)}: "
              + ", ".join(f"{o} {int((origin == o).sum())}" for o in sorted({str(o) for p in parts for o in p['origin']})))
    # precise-comb cut-offs, after the extra songs are added (so extra real songs count too)
    comb_cuts = [float(np.quantile(s[real], q)) for s in (s16, s44)]
    print(f"  precise-comb cut-offs (5% of real songs): 1-8 kHz {comb_cuts[0]:.2f}, 5-16 kHz {comb_cuts[1]:.2f}")

    if args.family:  # one boundary per generator family (multi-class); P(AI) = 1 - P(real)
        from family_test import fit_family
        from more44_test import FAMILY
        fam = np.array([FAMILY.get(str(s), str(s)) for s in labels])
        train = lambda use: fit_family(X, fam, use)  # noqa: E731
        print(f"One boundary per generator family: {len(set(fam)) - 1} families")
    else:
        train = lambda use: fit(X, y, use)  # noqa: E731

    print(f"Cross-validation for the cut-off (5 folds, {X.shape[1]} features) ...")
    oof = np.zeros(len(y))
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, labels, groups), 1):
        use = np.ones(len(y), bool)
        use[te] = False
        oof[te] = ai_proba(train(use), X[te])
        print(f"  fold {k} done")
    cut = float(np.quantile(oof[real], q))
    print(f"\nCut-off: P(AI) > {cut:.3f} flags {args.false_alarms:g}% of real songs "
          f"(cross-validated AUC {roc_auc_score(y, oof):.3f})")
    print("Cross-validated catch rate per generator (these generators are in training):")
    for g in sorted(set(labels[y])):
        print(f"  {g:12s} {100 * np.mean(oof[labels == g] > cut):5.1f}%  ({(labels == g).sum()} songs)")
    print(f"  {'ALL':12s} {100 * np.mean(oof[y] > cut):5.1f}%")
    if len(set(origin[real])) > 1:
        print("Cross-validated false alarms per real-song source:")
        for o in sorted(set(origin[real])):
            s = real & (origin == o)
            print(f"  {o:12s} {100 * np.mean(oof[s] > cut):5.1f}%  ({int(s.sum())} songs)")

    # Joint rule (as in twoband_test.py): each score -> share of real training songs below it; a song's
    # combined score is the highest of its three (classifier, comb 1-8 kHz, comb 5-16 kHz); one cut-off on
    # that flags 5% of real songs. The real-song scores are saved so predict can do the same.
    refs = {"clf": np.sort(oof[real]), "s16": np.sort(s16[real]), "s44": np.sort(s44[real])}
    joint = joint_score(refs, oof, s16, s44)
    joint_cut = float(np.quantile(joint[real], q))
    print(f"\nJoint rule (classifier or comb), cut-off {joint_cut:.4f} on the combined score; cross-validated catch rate:")
    for g in sorted(set(labels[y])):
        print(f"  {g:12s} {100 * np.mean(joint[labels == g] > joint_cut):5.1f}%")
    print(f"  {'ALL':12s} {100 * np.mean(joint[y] > joint_cut):5.1f}%")

    # Spacing bank (known_spacing_test.py): each generator's exact comb spacing, from the mean 5-16 kHz fakeprint of
    # its songs (training generators, plus --bank-extra songs used only for this); kept if that mean has a clear
    # comb. Each song's bank score = its highest known-comb score over the spacings, as the share of real songs below.
    n0 = len(bands[0]["freqs"])
    fh5 = np.asarray(bands[1]["freqs"], dtype=np.float64)
    groups_fp = {str(g): X[labels == g, n0:] for g in sorted(set(labels[y]))}
    from more44_test import FAMILY
    for f in [f for d in (args.bank_extra or []) for f in (sorted(Path(d).glob("*.npz")) if Path(d).is_dir() else [Path(d)])]:
        d = np.load(f)
        keep = d["src_sr"] >= bands[1]["sr"]
        for s in sorted(set(d["sources"][keep].astype(str))):
            g = FAMILY.get(s, s)
            rows = d["fp1"][keep & (d["sources"].astype(str) == s)].astype(np.float32)
            groups_fp[g] = np.vstack([groups_fp[g], rows]) if g in groups_fp else rows
    bank = {"spacings": [], "names": []}
    print(f"\nSpacing bank (mean-fakeprint comb >= {BANK_MIN:g}):")
    for g, rows in sorted(groups_fp.items()):
        m_sharp, s0 = precise_comb(fh5, rows.mean(0))
        same = [i for i, s in enumerate(bank["spacings"]) if abs(s - s0) < 0.02]
        print(f"  {g:12s} {m_sharp:5.1f} at {s0:8.3f} Hz ({len(rows)} songs)"
              + ("" if m_sharp < BANK_MIN else f"  -> same as {bank['names'][same[0]]}" if same else "  -> bank"))
        if m_sharp >= BANK_MIN:
            if same:
                bank["names"][same[0]] += f" / {g}"
            else:
                bank["spacings"].append(float(s0))
                bank["names"].append(str(g))
    ks = [known_comb(fh5, X[:, n0:], s) for s in bank["spacings"]]
    bank["real_scores"] = [np.sort(k[real]).astype(np.float32) for k in ks]
    pct = lambda ref, x: np.searchsorted(ref, x, side="right") / len(ref)  # noqa: E731
    b_all = np.max([pct(r, k) for r, k in zip(bank["real_scores"], ks)], 0)
    bank["real_bank"] = np.sort(b_all[real]).astype(np.float32)
    clf_p, bank_p = pct(refs["clf"], oof), pct(bank["real_bank"], b_all)
    for name, js in (("clf_bank", np.maximum(clf_p, bank_p)), ("clf_bank_comb", np.maximum(joint, bank_p))):
        bank[f"{name}_cutoff"] = float(np.quantile(js[real], q))
        bank[f"{name}_real"] = np.sort(js[real]).astype(np.float32)
        bank[f"{name}_ai"] = np.sort(js[y]).astype(np.float32)
        print(f"  {name}: cut-off {bank[f'{name}_cutoff']:.4f}; AI caught (the bank saw these songs, so high) "
              f"{100 * np.mean(js[y] > bank[f'{name}_cutoff']):.1f}%")
    bank["spacings"] = np.array(bank["spacings"])

    # Real songs with little high-frequency content (old recordings, low-bitrate files) may look
    # like low-band generators in the 5-16 kHz band: how often are they flagged?
    print("\nReal songs flagged, by how far up they have content:")
    for lo, hi in ((0, 12), (12, 16), (16, 99)):
        s = real & (top >= lo) & (top < hi)
        if s.any():
            print(f"  {lo:2d}-{hi if hi < 99 else '':>2} kHz {100 * np.mean(oof[s] > cut):5.1f}%  ({int(s.sum())} songs)")

    print("\nFinal model on all songs ...")
    model = train(np.ones(len(y), bool))
    joblib.dump({"model": model, "bands": bands, "max_duration": max_duration, "min_sr": bands[1]["sr"],
                 "cutoff": cut, "comb_cutoffs": comb_cuts, "n_features": X.shape[1],
                 "real_scores": {k: v.astype(np.float32) for k, v in refs.items()}, "joint_cutoff": joint_cut,
                 # cross-validated scores, sorted, for predict's percentile and base-rate estimate
                 "real_joint": np.sort(joint[real]).astype(np.float32),
                 "ai_scores": {"clf": np.sort(oof[y]).astype(np.float32), "joint": np.sort(joint[y]).astype(np.float32)},
                 "bank": bank,
                 "trained_on": {g: int((labels == g).sum()) for g in sorted(set(labels))}}, args.out)
    print(f"Saved {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

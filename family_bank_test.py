r"""Can a better default model come from parts we already have: LR per family + the spacing bank?

LR per family (one boundary per generator family) is the best classifier on generators it has seen and gives the
fewest false alarms, but it misses more brand-new generators (MiniMax 72% vs 81%): it asks "is this a family I
know?". The spacing bank (known_spacing_test.py) catches songs whose comb sits at a known decoder rate, and caught
97% of MiniMax although MiniMax is not in it. So the two may cover each other's gaps.
Four rules, all at 5% of real songs flagged (cut-offs from the real training songs):
  lr            LR one boundary alone (today's default, predict without options)
  lr+bank       LR one boundary or spacing bank (predict --bank)
  family        LR per family alone
  family+bank   LR per family or spacing bank
"or" = the highest of the two shares of real songs below (as the joint rule). The bank is always built only from
the AI songs the model trains on in that test (no peeking): each family's spacing from its mean 5-16 kHz fakeprint,
kept if that mean has a clear comb (precise comb >= 8).
Songs: stored 30 s fakeprints (features_tm\, as temporal_test.py): FMA small + Echoes + more44 + more_real at
44.1 kHz or more (16 AI families), live concerts held out. No download. Tests:
  cv     5-fold cross-validation (real songs grouped by artist)
  logo   leave one generator family out (it is also left out of the bank); half the held-out real fold sets the
         cut-off, half measures false alarms (as temporal_test.py)
  fresh  final models on all songs; fresh44 (Suno V5.5, MiniMax, Lyria 3, real), the 1,077 new real songs, the live
         concerts, and the robustness songs (speed, pitch, MP3, EQ ...), cut-offs from cv
Rule fixed before running: family+bank becomes predict's default if, compared with lr, it catches at least 2 points
more of fresh44's AI songs, loses at most 2 points on never-seen families (logo), and flags at most 6% of the
never-used real songs (fresh44's and the 1,077 new ones) and at most 0.5 points more than lr.

  python -I family_bank_test.py --quick 3000 --tests cv     # timing check
  python -I family_bank_test.py                             # ~1.5 hours; reports\family_bank_test.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold  # noqa: E402

import fakeprint as fpm  # noqa: E402
import temporal_test as tt  # noqa: E402

tt.SKIP |= {"tstd0", "tstd1", "tco0", "tco1", "shape0", "shape1", "stereo"}
HERE = Path(__file__).resolve().parent
BANK_MIN = 8.0
KINDS = ["lr", "family"]


def pct(ref, x):
    return np.searchsorted(ref, x, side="right") / len(ref)


class Bank:
    """Known-comb scores, cached by spacing (rounded to 0.001 Hz), for the training songs and every test set."""

    def __init__(self, f, sets):
        self.f, self.sets, self.cache = f, sets, {}

    def scores(self, s0):
        key = round(s0, 3)
        return _Lazy(self, key)

    def spacings(self, fp1, fam, rows):
        """The bank from the AI songs in `rows`: one spacing per family with a clear comb (merged within 0.02 Hz)."""
        out = []
        for g in sorted(set(fam[rows])):
            m_sharp, s0 = fpm.precise_comb(self.f, fp1[rows & (fam == g)].mean(0))
            if m_sharp >= BANK_MIN and not any(abs(s0 - s) < 0.02 for s in out):
                out.append(float(s0))
        return out

    def share(self, spacings, ref_rows, target):
        """Highest share of real songs (training rows ref_rows) below, over the bank spacings, for set `target`."""
        return np.max([pct(np.sort(self.scores(s)["train"][ref_rows]), self.scores(s)[target]) for s in spacings], 0)


class _Lazy:
    """bank.scores(s)[name]: the known-comb scores of one set at spacing s, computed on first use."""

    def __init__(self, bank, key):
        self.bank, self.key = bank, key

    def __getitem__(self, name):
        c = self.bank.cache.setdefault(self.key, {})
        if name not in c:
            c[name] = fpm.known_comb(self.bank.f, self.bank.sets[name], self.key)
        return c[name]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", nargs="+", default=["cv", "fresh", "logo"], choices=["cv", "fresh", "logo"])
    ap.add_argument("--kinds", nargs="+", default=KINDS, choices=KINDS)
    ap.add_argument("--quick", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "reports" / "family_bank_test.json"))
    args = ap.parse_args()
    out = Path(args.out)
    results = json.loads(out.read_text()) if out.exists() and not args.quick else {}
    print("Loading ...", flush=True)
    d, held, fr = tt.load(HERE / "features_tm", args.quick)
    for p in sorted((HERE / "features_tm" / "newreal").glob("*.npz")):
        x = np.load(p)
        ok = x["src_sr"] >= 44100
        held[p.stem] = {k: x[k][ok] for k in ("fp0", "fp1")}
    src = fr["sources"].astype(str)
    for s in sorted(set(src)):
        held[f"fresh44 {s}"] = {k: fr[k][src == s] for k in ("fp0", "fp1")}
    r = np.load(HERE / "features_hq" / "unseen" / "robust.npz")
    rreal = np.char.startswith(r["sources"].astype(str), "real")
    for c, cond in enumerate(r["conditions"].astype(str)):
        for who, m in (("AI", ~rreal), ("real", rreal)):
            held[f"robust {cond} | {who}"] = {"fp0": r["fp0"][c][m].astype(np.float32),
                                              "fp1": r["fp1"][c][m].astype(np.float32)}
    labels, groups, fam = d["sources"], d["groups"], d["fam"]
    real, y = labels == "real", labels != "real"
    X = np.hstack([d["fp0"], d["fp1"]]).astype(np.float32)
    Xh = {h: np.hstack([v["fp0"], v["fp1"]]).astype(np.float32) for h, v in held.items()}
    f5 = np.asarray(joblib.load(HERE / "model_twoband.joblib")["bands"][1]["freqs"], dtype=np.float64)
    bank = Bank(f5, {"train": d["fp1"].astype(np.float32)} | {h: v["fp1"].astype(np.float32) for h, v in held.items()})
    t0 = time.time()

    for kind in args.kinds:
        res = results.setdefault(kind, {})
        if any(t in args.tests and t not in res for t in ("cv", "fresh")):
            print(f"\n===== {kind}: cv =====", flush=True)
            oof, boof = np.zeros(len(y)), np.zeros(len(y))
            for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, labels, groups), 1):
                use = np.ones(len(y), bool)
                use[te] = False
                oof[te] = tt.train(kind, X, y, fam, use)(X[te])
                sp = bank.spacings(d["fp1"], fam, use & y)
                boof[te] = bank.share(sp, use & real, "train")[te]
                print(f"  fold {k}: {len(sp)} bank spacings ({(time.time() - t0) / 60:.1f} min)", flush=True)
            ref_clf = np.sort(oof[real])
            comb = np.maximum(pct(ref_clf, oof), boof)
            cut, jcut = float(np.quantile(oof[real], 0.95)), float(np.quantile(comb[real], 0.95))
            rows = {g: fam == g for g in sorted(set(fam[y]))} | {"ALL AI": y}
            rows |= {f"real {o}": real & (d["origin"] == o) for o in sorted(set(d["origin"][real]))}
            res["cv"] = {g: [float(np.mean(oof[m] > cut)), float(np.mean(comb[m] > jcut))] for g, m in rows.items()}
            if "fresh" in args.tests:
                print(f"===== {kind}: fresh =====", flush=True)
                score = tt.train(kind, X, y, fam, np.ones(len(y), bool))
                sp = bank.spacings(d["fp1"], fam, y)
                res["bank_spacings"] = sp
                res["fresh"] = {}
                for h in held:
                    p = score(Xh[h])
                    bs = np.max([pct(np.sort(bank.scores(s)["train"][real]), bank.scores(s)[h]) for s in sp], 0)
                    c2 = np.maximum(pct(ref_clf, p), bs)
                    res["fresh"][h] = [float(np.mean(p > cut)), float(np.mean(c2 > jcut)), int(len(p))]
            if not args.quick:
                out.write_text(json.dumps(results, indent=1))
        if "logo" in args.tests and "logo" not in res:
            print(f"\n===== {kind}: leave one family out =====", flush=True)
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
                score = tt.train(kind, X, y, fam, use)
                sp = bank.spacings(d["fp1"], fam, use & y)
                gi = np.flatnonzero(fam == g)
                ps, pg, pc = score(X[set_cut]), score(X[gi]), score(X[check])
                ps_sorted = np.sort(ps)  # pct() needs a sorted reference
                ref_rows = real.copy()  # bank shares against all real songs except the check half (no fitting
                ref_rows[check] = False  # on real songs, so this is fair); a small reference set saturates at 1.0
                bshare = bank.share(sp, ref_rows, "train")

                def comb(pr, ii):
                    return np.maximum(pct(ps_sorted, pr), bshare[ii])
                c1, c2 = np.quantile(ps, 0.95), np.quantile(comb(ps, set_cut), 0.95)
                res["logo"][g] = [float(np.mean(pg > c1)), float(np.mean(comb(pg, gi) > c2)), int(len(gi))]
                fa.append([float(np.mean(pc > c1)), float(np.mean(comb(pc, check) > c2))])
                print(f"  left out {g:12s} {100 * res['logo'][g][0]:5.1f}% / with bank {100 * res['logo'][g][1]:5.1f}%"
                      f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
            n = np.array([v[2] for v in res["logo"].values()])
            res["logo"]["ALL AI"] = [float(n @ [v[j] for v in res["logo"].values()] / n.sum()) for j in (0, 1)] + [int(n.sum())]
            res["logo"]["REAL flagged"] = [float(np.mean([a[0] for a in fa])), float(np.mean([a[1] for a in fa]))]
            if not args.quick:
                out.write_text(json.dumps(results, indent=1))
    summary(results)


def summary(results):
    def pooled(res, names, j):
        n = np.array([res["fresh"][x][2] for x in names])
        return float(n @ [res["fresh"][x][j] for x in names] / n.sum())
    table, cols = {}, []
    for kind, res in results.items():
        for j, name in ((0, kind), (1, f"{kind}+bank")):
            cols.append(name)
            t = table.setdefault(name, {})
            if "cv" in res:
                t["AI caught, cv"] = res["cv"]["ALL AI"][j]
            if "logo" in res:
                t["AI caught, family never seen"] = res["logo"]["ALL AI"][j]
                t["real flagged, logo"] = res["logo"]["REAL flagged"][j]
            if "fresh" in res:
                fs = res["fresh"]
                ai = [x for x in fs if x.startswith("fresh44") and "real" not in x]
                nr = [x for x in fs if x.startswith("fresh44 real") or x in ("community", "jamendo", "mixtapes")]
                t["fresh44 AI caught"] = pooled(res, ai, j)
                for x in ai:
                    t[x] = fs[x][j]
                t["never-used real flagged"] = pooled(res, nr, j)
                t["live concerts flagged"] = fs["etree"][j]
                t["speed changes AI (mean)"] = float(np.mean([fs[f"robust speed {s} | AI"][j]
                                                              for s in ("-3%", "-1%", "+1%", "+3%")]))
            if "cv" in res:
                for g in ("mureka", "sonauto", "lyria", "udio", "elevenlabs"):
                    if g in res["cv"]:
                        t[f"{g}, cv"] = res["cv"][g][j]
            if "logo" in res:
                for g in ("mureka", "sonauto", "lyria", "udio", "heartmula", "suno"):
                    if g in res["logo"]:
                        t[f"{g}, never seen"] = res["logo"][g][j]
    rows = list(dict.fromkeys(r for t in table.values() for r in t))
    print(f"\n{'':32s}" + "".join(f"{c:>14s}" for c in cols))
    for r in rows:
        print(f"{r:32s}" + "".join(f"{100 * table[c][r]:13.1f}%" if r in table[c] else f"{'-':>14s}" for c in cols))
    a, b = table.get("family+bank", {}), table.get("lr", {})
    need = ("fresh44 AI caught", "AI caught, family never seen", "never-used real flagged")
    if all(k in a and k in b for k in need):
        ok = (a[need[0]] - b[need[0]] >= 0.02 and a[need[1]] - b[need[1]] >= -0.02
              and a[need[2]] <= 0.06 and a[need[2]] - b[need[2]] <= 0.005)
        print(f"\nRule (family+bank vs lr): fresh44 AI {100 * (a[need[0]] - b[need[0]]):+.1f} points, never-seen families "
              f"{100 * (a[need[1]] - b[need[1]]):+.1f}, never-used real flagged {100 * a[need[2]]:.1f}% "
              f"({100 * (a[need[2]] - b[need[2]]):+.1f}) -> {'ADOPT' if ok else 'KEEP lr'}")


if __name__ == "__main__":
    main()

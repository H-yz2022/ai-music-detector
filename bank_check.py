r"""Check the spacing bank of model_twoband.joblib on songs it never saw, before it goes into `predict --joint`.

Three joint rules, all with cut-offs from the real training songs (5% flagged, train_twoband.py):
  A  classifier or precise comb (1-8 and 5-16 kHz): today's --joint
  B  classifier or spacing bank
  C  classifier or precise comb or spacing bank
(and the classifier alone, for reference). Songs, all never trained on and not used to build the bank:
  fresh44: Suno V5.5, MiniMax (689 Hz: not in the bank), Lyria 3, and 377 real songs
  the 1,077 new real songs (features_tm\newreal\)
  the robustness songs (features_hq\unseen\robust.npz): every change (speed, pitch, MP3, EQ ...), AI and real
Stored 30 s fakeprints, no download.
Rule fixed before running: B or C replaces A in --joint, whichever catches more of fresh44's AI songs, if it
(1) flags at most 6% of the never-used real songs (fresh44's and the new ones) and (2) catches, on average over the
four speed changes, at most 5 points fewer of the robustness AI songs than A. Otherwise A stays.

  python -I bank_check.py        # ~2 min; reports\bank_check.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

import fakeprint as fpm  # noqa: E402
from detect import ai_proba, joint_score  # noqa: E402

HERE = Path(__file__).resolve().parent


def pct(ref, x):
    return np.searchsorted(ref, x, side="right") / len(ref)


def rules(b, fp0, fp1, s16, s44):
    """Flags (True = AI) of each rule for songs with these fakeprints and precise combs."""
    f5 = np.asarray(b["bands"][1]["freqs"], dtype=np.float64)
    p = ai_proba(b["model"], np.hstack([fp0, fp1]).astype(np.float32))
    bank = b["bank"]
    bs, _ = fpm.bank_score(f5, fp1, bank)
    bank_p = pct(bank["real_bank"], bs)
    clf_p = pct(b["real_scores"]["clf"], p)
    joint = joint_score(b["real_scores"], p, s16, s44)
    return {"clf": p > b["cutoff"], "A": joint > b["joint_cutoff"],
            "B": np.maximum(clf_p, bank_p) > bank["clf_bank_cutoff"],
            "C": np.maximum(joint, bank_p) > bank["clf_bank_comb_cutoff"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(HERE / "model_twoband.joblib"))
    ap.add_argument("--out", default=str(HERE / "reports" / "bank_check.json"))
    args = ap.parse_args()
    b = joblib.load(args.model)
    sets = {}
    f = np.load(HERE / "features_tm" / "fresh44.npz")
    hi = f["src_sr"] >= 44100
    src = f["sources"].astype(str)
    for s in sorted(set(src[hi])):
        m = hi & (src == s)
        sets[f"fresh44 {s}"] = (f["fp0"][m], f["fp1"][m], f["sharp0"][m], f["sharp1"][m])
    for p in sorted((HERE / "features_tm" / "newreal").glob("*.npz")):
        x = np.load(p)
        m = x["src_sr"] >= 44100
        sets[f"new real {p.stem}"] = (x["fp0"][m], x["fp1"][m], x["sharp0"][m], x["sharp1"][m])
    r = np.load(HERE / "features_hq" / "unseen" / "robust.npz")
    rreal = np.char.startswith(r["sources"].astype(str), "real")
    for c, cond in enumerate(r["conditions"].astype(str)):
        for who, m in (("AI", ~rreal), ("real", rreal)):
            sets[f"robust {cond} | {who}"] = (r["fp0"][c][m].astype(np.float32), r["fp1"][c][m].astype(np.float32),
                                              r["sharp0"][c][m], r["sharp1"][c][m])
    res = {}
    for name, (a0, a1, s16, s44) in sets.items():
        fl = rules(b, a0, a1, s16, s44)
        res[name] = {k: [float(v.mean()), int(len(v))] for k, v in fl.items()}

    def pooled(names, k):
        n = np.array([res[x][k][1] for x in names])
        return float(n @ [res[x][k][0] for x in names] / n.sum())
    ai = [x for x in res if x.startswith("fresh44") and "real" not in x]
    real = [x for x in res if x.startswith("fresh44 real") or x.startswith("new real")]
    speed = [f"robust speed {s} | AI" for s in ("-3%", "-1%", "+1%", "+3%")]
    summary = {k: {"fresh44 AI": pooled(ai, k), "never-used real": pooled(real, k),
                   "speed changes AI (mean)": float(np.mean([res[x][k][0] for x in speed]))}
               for k in ("clf", "A", "B", "C")}
    print(f"{'':34s}" + "".join(f"{k:>9s}" for k in ("clf", "A", "B", "C")))
    for name, v in res.items():
        print(f"{name:34s}" + "".join(f"{100 * v[k][0]:8.1f}%" for k in ("clf", "A", "B", "C")) + f"   ({v['A'][1]})")
    print()
    for row in ("fresh44 AI", "never-used real", "speed changes AI (mean)"):
        print(f"{row:34s}" + "".join(f"{100 * summary[k][row]:8.1f}%" for k in ("clf", "A", "B", "C")))
    ok = [k for k in ("B", "C") if summary[k]["never-used real"] <= 0.06
          and summary[k]["speed changes AI (mean)"] >= summary["A"]["speed changes AI (mean)"] - 0.05]
    pick = max(ok, key=lambda k: summary[k]["fresh44 AI"]) if ok else "A"
    print(f"\nRule: passes {ok or 'none'} -> --joint uses {pick}")
    Path(args.out).write_text(json.dumps({"sets": res, "summary": summary, "pick": pick},
                                                                 indent=1))


if __name__ == "__main__":
    main()

r"""Whole-song cut-off and real-song ranks for detect.py predict (wholesong_test.py: needs its own cut-off).

The model (model_twoband.joblib) stays trained on 30 s clips. Averaging a whole song gives a cleaner fakeprint, so
every song scores higher than its 30 s clip, and the 30 s cut-off would flag ~14% of real songs. This scores 1,130
complete real songs (whole song, up to 8 min, from features_tm\wholesong\: the new real songs of newreal_test.py and
fresh44's Jamendo songs, all at 44.1 kHz or more and 60 s or longer) and saves, in model_twoband.joblib["whole"]:
  real_scores   their classifier and precise-comb scores (sorted), for the ranks and the joint rule
  real_joint    their joint scores (sorted)
  cutoff, joint_cutoff   flag 5% of them
  ai_scores     classifier and joint scores of fresh44's complete AI songs (Suno V5.5, MiniMax, Lyria 3), sorted,
                for --base-rate
Checked on a new test set by wholecheck_test.py.

  python -I calibrate_whole.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from detect import ai_proba, joint_score  # noqa: E402

HERE = Path(__file__).resolve().parent
REAL = {"community", "jamendo", "mixtapes", "real_jamendo"}


def main():
    path = HERE / "model_twoband.joblib"
    b = joblib.load(path)
    parts = [dict(np.load(HERE / "features_tm" / "wholesong" / f"{w}.npz")) for w in ("newreal", "fresh44")]
    d = {k: np.concatenate([p[k] for p in parts]) for k in ("names", "sources", "fp0", "fp1", "sharp0", "sharp1",
                                                            "src_sr", "seconds")}
    src = d["sources"].astype(str)
    ok = (d["src_sr"] >= b["min_sr"]) & (d["seconds"] >= 60)
    p = ai_proba(b["model"], np.hstack([d["fp0"], d["fp1"]]))
    real, ai = ok & np.isin(src, list(REAL)), ok & ~np.isin(src, list(REAL))
    refs = {"clf": np.sort(p[real]), "s16": np.sort(d["sharp0"][real]), "s44": np.sort(d["sharp1"][real])}
    joint = joint_score(refs, p, d["sharp0"], d["sharp1"])
    w = {"real_scores": {k: v.astype(np.float32) for k, v in refs.items()},
         "real_joint": np.sort(joint[real]).astype(np.float32),
         "cutoff": float(np.quantile(p[real], 0.95)), "joint_cutoff": float(np.quantile(joint[real], 0.95)),
         "ai_scores": {"clf": np.sort(p[ai]).astype(np.float32), "joint": np.sort(joint[ai]).astype(np.float32)},
         "min_seconds": 60.0, "max_seconds": 480.0,
         "calibrated_on": {s: int((ok & (src == s)).sum()) for s in sorted(set(src))}}
    b["whole"] = w
    joblib.dump(b, path)
    print(f"{int(real.sum())} real songs, {int(ai.sum())} AI songs (whole song, median {np.median(d['seconds'][ok]):.0f} s)")
    print(f"Whole-song cut-off: P(AI) > {w['cutoff']:.3f} (30 s: {b['cutoff']:.3f}); joint > {w['joint_cutoff']:.4f}")
    print(f"AI songs above the cut-off: {100 * np.mean(p[ai] > w['cutoff']):.1f}% (with comb "
          f"{100 * np.mean(joint[ai] > w['joint_cutoff']):.1f}%; these songs were tested before, so only a sanity check)")
    print(f"Saved in {path.name}['whole']")


if __name__ == "__main__":
    main()

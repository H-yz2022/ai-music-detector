r"""Held-out check: pick settings on one sample ("dev"), then test them on a fresh one.

- Paper's classifier: trained on the dev sample only, scored on the fresh sample.
- Comb score: old rule vs new rule (bump removal, 15 Hz wide, chosen on dev).
  Each rule's threshold is set on dev so that 5% of dev real songs are flagged,
  then used unchanged on the fresh sample. Only the fresh sample says whether
  the new rule really helps.

  python -I evaluate.py                        # dev = data, fresh = data_fresh
  python -I evaluate.py --dev data --test data_fresh --max-duration 30
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import fakeprint as fpm  # noqa: E402
from detect import PAPER, featurize_folder, make_model  # noqa: E402
from report import write_report  # noqa: E402

HERE = Path(__file__).resolve().parent
GENS = ("real", "suno", "udio")


def load_set(root, args):
    band = (args.sr, args.fmin, args.fmax, args.max_duration)
    items = featurize_folder(root / "real", "real", *band) + featurize_folder(root / "ai", "ai", *band)
    for it in items:
        m = re.search(r"_(suno|udio)_", it["name"].lower())
        it["gen"] = m.group(1) if m else ("real" if it["group"] == "real" else "ai")
        it["comb_old"], it["sp_old"] = fpm.comb_score(it["freqs"], it["fp"])
        it["comb_new"], it["sp_new"] = fpm.comb_score(it["freqs"], it["fp"], bump_hz=args.bump_hz)
    return items


def rate(items, gen, hit):
    sel = [it for it in items if it["gen"] == gen]
    return 100 * np.mean([hit(it) for it in sel]) if sel else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", default=str(HERE / "data"))
    ap.add_argument("--test", default=str(HERE / "data_fresh"))
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--fmin", type=float, default=1000.0)
    ap.add_argument("--fmax", type=float, default=8000.0)
    ap.add_argument("--max-duration", type=float, default=30.0)
    ap.add_argument("--bump-hz", type=float, default=fpm.BUMP_HZ)
    ap.add_argument("--false-alarms", type=float, default=5.0, help="%% of dev real songs a comb rule may flag")
    ap.add_argument("--out", default=str(HERE / "reports" / "fresh_test.html"))
    args = ap.parse_args()

    dev, test = load_set(Path(args.dev), args), load_set(Path(args.test), args)
    model = make_model().fit(np.stack([it["fp"] for it in dev]), [int(it["group"] == "ai") for it in dev])
    probs = model.predict_proba(np.stack([it["fp"] for it in test]))[:, 1]
    for it, p in zip(test, probs):
        it["prob"] = float(p)

    # Comb thresholds come from the dev real songs only.
    thr = {k: float(np.quantile([it[k] for it in dev if it["gen"] == "real"], 1 - args.false_alarms / 100))
           for k in ("comb_old", "comb_new")}
    names = {"comb_old": "Comb, old", "comb_new": f"Comb, bumps removed ({args.bump_hz:g} Hz)"}
    rows = [("Paper's classifier, fresh", lambda it: it["prob"] >= 0.5, test)]
    for key, label in names.items():
        rows.append((f"{label}, dev", lambda it, k=key: it[k] > thr[k], dev))
        rows.append((f"{label}, fresh", lambda it, k=key: it[k] > thr[k], test))

    print(f"\nComb thresholds (set on dev, {args.false_alarms:g}% of dev real flagged): "
          + ", ".join(f"{names[k]} {v:.3f}" for k, v in thr.items()))
    print(f"\n{'':40s}{'real flagged':>14s}{'Suno caught':>13s}{'Udio caught':>13s}")
    for label, hit, items in rows:
        r, s, u = (rate(items, g, hit) for g in GENS)
        print(f"{label:40s}{r:13.0f}%{s:12.0f}%{u:12.0f}%")

    def auc(items, key, gen):
        sel = [it for it in items if it["gen"] in ("real", gen)]
        return roc_auc_score([it["gen"] == gen for it in sel], [it[key] for it in sel])

    y_test = [it["group"] == "ai" for it in test]
    print(f"\nAUC, paper's classifier (real vs all AI), fresh: {roc_auc_score(y_test, probs):.3f}")
    for key in ("comb_old", "comb_new"):
        for gen in ("suno", "udio"):
            print(f"AUC, {names[key]}, real vs {gen.capitalize()}: "
                  f"dev {auc(dev, key, gen):.3f}  fresh {auc(test, key, gen):.3f}")

    # Report for the fresh sample, using the new comb rule.
    t = thr["comb_new"]
    for it in test:
        it["group"], it["comb_score"], it["spacing_hz"] = it["gen"], it["comb_new"], it["sp_new"]
    test.sort(key=lambda it: GENS.index(it["gen"]) if it["gen"] in GENS else 9)
    cards = [
        ("Classifier: real called real", f"{100 - rate(test, 'real', lambda it: it['prob'] >= 0.5):.0f}%",
         "trained on dev sample, tested on fresh"),
        ("Classifier: Suno / Udio caught",
         f"{rate(test, 'suno', lambda it: it['prob'] >= 0.5):.0f}% / {rate(test, 'udio', lambda it: it['prob'] >= 0.5):.0f}%",
         "fresh songs it never saw"),
        ("Comb (bumps removed): real flagged", f"{rate(test, 'real', lambda it: it['comb_score'] > t):.0f}%",
         f"threshold {t:.3f}, set on dev"),
        ("Comb (bumps removed): Suno caught", f"{rate(test, 'suno', lambda it: it['comb_score'] > t):.0f}%",
         "no training"),
    ]
    intro = (f"Held-out test of {PAPER} fakeprints. Settings were chosen on a first sample (dev) and are scored here "
             f"on fresh songs: {sum(it['gen'] == 'real' for it in test)} real FMA clips, "
             f"{sum(it['gen'] == 'suno' for it in test)} Suno and {sum(it['gen'] == 'udio' for it in test)} Udio songs, "
             f"first {args.max_duration:g} s, {args.sr / 1000:g} kHz, {args.fmin / 1000:g}–{args.fmax / 1000:g} kHz band.")
    path = write_report(args.out, "AI Music Detector · Fresh-sample test", intro, test, cards, t,
                        coef=model.coef_[0],
                        notes=["P(AI) comes from the classifier trained on the dev sample only.",
                               f"Comb scores use bump removal ({args.bump_hz:g} Hz running mean). "
                               f"The threshold flags {args.false_alarms:g}% of dev real songs."])
    print(f"Report: {path}")


if __name__ == "__main__":
    main()

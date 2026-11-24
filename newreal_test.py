r"""New real songs, never used before: does LightGBM + stereo (temporal_test.py) flag too many of them?

temporal_test.py found LightGBM + stereo + comb the best candidate (92% cross-validated, 68% on never-seen families,
95% MiniMax, 75% Lyria 3), but it flagged 6.9% of fresh44's real songs and was picked after looking at fresh44. This
checks it on real songs from three Internet Archive collections never used here:
  jamendo     Jamendo albums (Creative Commons; many genres)
  mixtapes    hip-hop mixtapes (commercial-style production; a genre FMA small has few songs of)
  community   Community Audio uploads tagged "music"
Only items added before 2022 (before AI song generators were public), not in the netlabels or etree collections,
one song per artist, 30 s from 40% into the song, files at 44.1 kHz or more. Features as temporal_build.py (fakeprints
+ shape, stereo, time), no audio kept. Download: --build (~1.3 GB, ~25 min).
Models are trained on all training songs (as temporal_test.py), cut-offs from cross-validation (5% of real training
songs), then scored on the new songs. Compared: LR per family and LR one boundary (fakeprint), LightGBM (fakeprint),
LightGBM + stereo; each alone and with the precise comb (joint).
Rule fixed before running: LightGBM + stereo + comb passes if it flags at most 6% of all new real songs, and at most
1 point more than LR per family + comb.

  python -I newreal_test.py --build --n 3     # quick trial of the download (3 songs per source, scratch folder)
  python -I newreal_test.py --build           # ~1.3 GB streamed -> features_tm\newreal\
  python -I newreal_test.py                   # ~1 hour; reports\newreal_test.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
NOT_USED = ' AND format:"VBR MP3" AND addeddate:[* TO 2021-12-31] AND NOT collection:netlabels AND NOT collection:etree'
SOURCES = {"jamendo": "collection:jamendo-albums" + NOT_USED,
           "mixtapes": "collection:hiphopmixtapes" + NOT_USED,
           "community": "collection:opensource_audio AND subject:music" + NOT_USED}
RUNS = [("family", "avg"), ("lr", "avg"), ("lgbm", "avg"), ("lgbm", "avg+stereo")]


def build(out_dir, n, workers):
    import fresh44_test as f44
    import more_real
    import temporal_build as tb
    more_real.IA_SOURCES = SOURCES
    more_real.TITLE_ARTIST = {"jamendo"}
    more_real.fma_part = lambda n, seed: ([], {}, 0)  # no FMA songs this time
    more_real.quiet_worker = tb.init_worker  # workers compute temporal_build's features (with stereo)
    more_real.song_features = tb.song_features_tm
    f44.song_features = tb.song_features_tm
    sys.argv = ["more_real.py", "--out-dir", str(out_dir), "--fma", "0", "--ia", "400", "--budget-gb", "1.6",
                "--workers", str(workers), "--seed", "1"] + (["--n", str(n)] if n else [])
    more_real.main()
    g = out_dir / "fma_genres.json"
    if g.exists():
        g.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--features", default=str(HERE / "features_tm"))
    ap.add_argument("--new", default=str(HERE / "features_tm" / "newreal"))
    ap.add_argument("--out", default=str(HERE / "reports" / "newreal_test.json"))
    args = ap.parse_args()
    if args.build:
        build(Path(args.new), args.n, args.workers)
        return

    import temporal_test as tt
    from nonlinear_test import NAMES
    print("Loading ...")
    d, held, f = tt.load(Path(args.features), 0)
    for p in sorted(Path(args.new).glob("*.npz")):
        x = dict(np.load(p))
        ok = x["src_sr"] >= 44100
        held[p.stem] = {k: v[ok] for k, v in x.items() if getattr(v, "ndim", 0) and len(v) == len(ok)}
        mono = float(np.mean(np.all(held[p.stem]["stereo"] <= -59.9, axis=1)))
        print(f"  new real songs, {p.stem}: {int(ok.sum())} of {len(ok)} at 44.1 kHz or more ({100 * mono:.0f}% mono)")
    new = [h for h in held if h != "etree"]
    X, Xh, Xf = tt.matrices(d, held, f)
    out = Path(args.out)
    results = json.loads(out.read_text()) if out.exists() else {}
    for kind, s in RUNS:
        key = f"{kind} {s}"
        if key in results:
            continue
        print(f"\n===== {NAMES[kind]}, {s} =====", flush=True)
        r = tt.evaluate(kind, X[s], d, {h: v[s] for h, v in Xh.items()}, held, Xf[s], f, ["cv", "fresh"])
        n = np.array([r["fresh"][f"held {h}"][2] for h in new])
        r["fresh"]["ALL NEW REAL"] = [float(n @ [r["fresh"][f"held {h}"][j] for h in new] / n.sum()) for j in (0, 1)] \
            + [int(n.sum())]
        results[key] = r
        out.write_text(json.dumps(results, indent=1))

    rows = [f"held {h}" for h in new] + ["ALL NEW REAL", "held etree", "ALL REAL"]
    print(f"\nReal songs flagged: classifier (with comb)\n  {'':16s}" + "".join(f"{k:>20s}" for k in results))
    for row in rows:
        print(f"  {row.replace('held ', ''):16s}" + "".join(
            f"{100 * results[k]['fresh'][row][0]:11.1f}% ({100 * results[k]['fresh'][row][1]:4.1f}%)" for k in results))
    a, b = results["lgbm avg+stereo"]["fresh"]["ALL NEW REAL"][1], results["family avg"]["fresh"]["ALL NEW REAL"][1]
    ok = a <= 0.06 and a <= b + 0.01
    print(f"\nRule: LightGBM + stereo + comb flags {100 * a:.1f}% of new real songs (LR per family + comb: "
          f"{100 * b:.1f}%) -> {'PASSES' if ok else 'FAILS'}")


if __name__ == "__main__":
    main()

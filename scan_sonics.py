r"""Training-free comb scan: real songs vs Suno vs Udio, with an HTML report.

SONICS file names look like fake_50960_suno_0.mp3, so the generator (suno or
udio) is read from the name. SONICS audio is 16 kHz, so the defaults use the
paper's SONICS band of 1-8 kHz.

  python -I scan_sonics.py                                     # data\ai\sonics only
  python -I scan_sonics.py --real data\real\fma16k --max-duration 30
"""
import argparse
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import fakeprint as fpm  # noqa: E402
from audio_io import list_audio_files, load_audio  # noqa: E402
from detect import COMB_THRESHOLD, PAPER  # noqa: E402
from report import write_report  # noqa: E402

HERE = Path(__file__).resolve().parent
ORDER = {"real": 0, "suno": 1, "udio": 2}


def scan(folder, group, args):
    items = []
    for path in list_audio_files(folder):
        m = re.search(r"_(suno|udio)_", path.name.lower())
        audio, sr = load_audio(path, target_sr=args.sr, max_duration=args.max_duration)
        g = group or (m.group(1) if m else "ai")
        items.append({"name": path.name, "group": g, **fpm.extract(audio, sr, args.fmin, args.fmax)})
        print(f"  {g:4s} {path.name}  comb={items[-1]['comb_score']:.2f}")
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("songs", nargs="?", default=str(HERE / "data" / "ai" / "sonics"))
    ap.add_argument("--real", nargs="*", default=[], help="folder(s) of human-made songs")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--fmin", type=float, default=1000.0)
    ap.add_argument("--fmax", type=float, default=8000.0)
    ap.add_argument("--max-duration", type=float, default=180.0, help="seconds analysed per song")
    ap.add_argument("--out", default=str(HERE / "reports" / "sonics_scan.html"))
    args = ap.parse_args()

    items = scan(args.songs, None, args)
    for folder in args.real:
        items += scan(folder, "real", args)
    if not items:
        sys.exit("No audio files found.")
    items.sort(key=lambda it: ORDER.get(it["group"], 9))

    cards = []
    for g in sorted({it["group"] for it in items}, key=lambda g: ORDER.get(g, 9)):
        members = [it for it in items if it["group"] == g]
        found = [it for it in members if it["comb_score"] >= COMB_THRESHOLD]
        label = "false alarms" if g == "real" else "comb found"
        cards.append((f"{g.capitalize()}: {label}", f"{len(found)} / {len(members)}",
                      f"median score {np.median([it['comb_score'] for it in members]):.2f}"))
        top = collections.Counter(round(it["spacing_hz"]) for it in found).most_common(1)
        if top and g != "real":
            cards.append((f"{g.capitalize()}: most common spacing", f"{top[0][0]} Hz",
                          f"{top[0][1]} of {len(found)} songs with a comb"))

    intro = (f"Training-free comb scan ({PAPER} fakeprints plus a regular-spacing test). "
             f"Settings: {args.sr / 1000:g} kHz, {args.fmin / 1000:g}–{args.fmax / 1000:g} kHz band, "
             f"first {args.max_duration:g} s of each song, comb threshold {COMB_THRESHOLD}. "
             "A comb of evenly spaced spectral peaks points to a transposed-convolution decoder.")
    notes = []
    if not args.real:
        notes.append("<b>No real songs are included</b>, so this shows what AI songs look like, not the "
                     "false-alarm rate.")
    else:
        notes.append("Real songs are Creative Commons clips from FMA, converted to the same 16 kHz mono "
                     "40 kbps MP3 format as the SONICS AI songs, so the file format isn't a giveaway.")
    path = write_report(args.out, "AI Music Detector · Real vs Suno vs Udio", intro, items, cards,
                        COMB_THRESHOLD, notes=notes)
    print("\n".join(f"  {l}: {v}  ({s})" for l, v, s in cards))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()

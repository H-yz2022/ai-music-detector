r"""Example songs for the web demo (app.py): one human song, one Suno song, one MiniMax song, 60 s each.

Candidates, all under licenses that allow sharing with credit:
  human    the 13 Jamendo songs in jamendolyrics/jam-alt under CC BY or CC BY-SA
  suno     8 random Suno V5.5 songs (Kukedlc/suno-ai-music-dataset, CC BY 4.0)
  minimax  8 random MiniMax Music 3.0 songs (AngelSoftware/minimax-music-3-dataset, CC BY 4.0; never trained on)
The first ~60 s of each is streamed to a temporary folder and scored exactly as the web demo does (demo.analyze).
From the candidates that get the right answer (human: likely real; AI: likely AI), the one closest to the
median rank of its group is kept (a typical case, not the most extreme one), saved in examples\ with its credit
in examples\credits.json. The other clips are deleted. If no candidate of a group gets the right answer, that
group is left out (and said so).

  python -I make_examples.py      # ~45 MB streamed, ~4 MB kept
"""
import argparse
import csv
import io
import json
import random
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import demo  # noqa: E402
import fresh44_test as f44  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "examples"
f44.SECONDS = 60.0


def get(url):
    r = urllib.request.Request(url, headers=f44.UA)
    return urllib.request.urlopen(r, timeout=60).read().decode("utf-8", "replace")


def candidates(seed=0, titles=None):
    meta = list(csv.DictReader(io.StringIO(get(
        "https://huggingface.co/datasets/jamendolyrics/jam-alt/resolve/main/metadata.csv"))))
    ok = {r["Filepath"]: r for r in meta if r["LicenseType"] in ("CC BY", "CC BY-SA")}
    out = []
    for _, name, (_, url) in f44.hf_jobs("real_jamendo", "jamendolyrics/jam-alt", "subsets", 0, 0, seed):
        r = ok.get(name.split("/", 1)[1])
        if r:
            out.append(("human", url, {"title": r["Title"], "artist": r["Artist"], "license": r["LicenseType"],
                                       "link": r["URL"], "source": "Jamendo (via jamendolyrics/jam-alt)"}))
    for group, repo, folder in (("suno", "Kukedlc/suno-ai-music-dataset", "audio"),
                                ("minimax", "AngelSoftware/minimax-music-3-dataset", "audio/mp3")):
        for _, name, (_, url) in f44.hf_jobs(group, repo, folder, 1_000_000, 8, seed):
            out.append((group, url, {"title": Path(name).stem, "license": "CC BY 4.0",
                                     "link": f"https://huggingface.co/datasets/{repo}",
                                     "source": f"{'Suno V5.5' if group == 'suno' else 'MiniMax Music 3.0'} "
                                               f"(Hugging Face: {repo})"}))
    keep = lambda c: c[0] == "minimax" or not titles or c[2]["title"] in titles  # noqa: E731
    return [c for c in out if keep(c)]


BITRATES = {1: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
            2: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]}
RATES = {3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000]}


def fix_length(b):
    """A cut MP3 keeps the full song's frame count in its Xing / Info header, so players show the whole song's
    length. Set the count (and byte size) to what is left, and drop a cut-off last frame. The header frame holds
    no audio, and the decoded audio stays identical (checked when the examples were made)."""
    i = 0
    if b[:3] == b"ID3":
        s = b[6:10]
        i = 10 + ((s[0] << 21) | (s[1] << 14) | (s[2] << 7) | s[3])
    fr = []
    while i + 4 <= len(b):
        h = b[i:i + 4]
        if not (h[0] == 0xFF and (h[1] & 0xE0) == 0xE0):
            break
        ver, bi, si = (h[1] >> 3) & 3, h[2] >> 4, (h[2] >> 2) & 3
        if ver == 1 or bi in (0, 15) or si == 3:
            break
        n = (144 if ver == 3 else 72) * BITRATES[1 if ver == 3 else 2][bi] * 1000 // RATES[ver][si] + ((h[2] >> 1) & 1)
        if i + n > len(b):
            break
        fr.append((i, n))
        i += n
    if not fr:
        return b
    first = b[fr[0][0]:fr[0][0] + fr[0][1]]
    k = max(first.find(b"Xing"), first.find(b"Info"))
    if k < 0:
        return b
    flags = int.from_bytes(first[k + 4:k + 8], "big")
    end = fr[-1][0] + fr[-1][1]
    out = bytearray(b[:end])
    pos = fr[0][0] + k + 8
    if flags & 1:
        out[pos:pos + 4] = (len(fr) - 1).to_bytes(4, "big")
        pos += 4
    if flags & 2:
        out[pos:pos + 4] = (end - fr[0][0]).to_bytes(4, "big")
    return bytes(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--titles", nargs="+", default=None,
                    help="only these human / Suno candidates (MiniMax: all 8), e.g. after a first run")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="examples_"))
    two, one = demo.load_bundle("twoband"), demo.load_bundle("family")
    rows, total = [], 0
    try:
        for k, (group, url, credit) in enumerate(candidates(titles=args.titles)):
            try:
                blob, n = f44.mp3_start(url)
            except Exception as exc:
                print(f"  skipped ({exc})")
                continue
            total += n
            path = tmp / f"{group}_{k:02d}.mp3"
            path.write_bytes(blob)
            try:
                r = demo.analyze(path, two, one)
            except Exception as exc:
                print(f"  {path.name}: could not score ({exc})")
                continue
            if r["rank"] is None:  # below 44.1 kHz: the simpler 16 kHz model, no rank; not used as an example
                print(f"  {group:8s} {r['sample_rate']} Hz: below 44.1 kHz, left out")
                continue
            rows.append((group, path, credit, r))
            print(f"  {group:8s} {r['level']:12s} rank {100 * r['rank']:5.1f}%  {r['sample_rate']} Hz  "
                  f"{r.get('bank_like', '') or ''}  {credit['title'][:40]}", flush=True)
        print(f"Streamed {total / 1e6:.0f} MB")
        OUT.mkdir(exist_ok=True)
        credits = {}
        for group, want in (("human", "likely real"), ("suno", "likely AI"), ("minimax", "likely AI")):
            g = [x for x in rows if x[0] == group]
            good = [x for x in g if x[3]["level"] == want]
            print(f"{group}: {len(good)} of {len(g)} get the right answer ({want})")
            if not good:
                continue
            med = np.median([x[3]["rank"] for x in g])
            group_, path, credit, r = min(good, key=lambda x: abs(x[3]["rank"] - med))
            dest = OUT / f"{group}.mp3"
            dest.write_bytes(fix_length(path.read_bytes()))
            import soundfile as sf
            credits[group] = credit | {"file": dest.name, "seconds": round(sf.info(str(dest)).duration),
                                       "scored_seconds": round(r["seconds"]), "rank": round(r["rank"], 4),
                                       "chosen_from": f"{len(good)} of {len(g)} candidates got '{want}'"}
            print(f"  kept {dest.name}: {credit['title']} ({100 * r['rank']:.1f}%, {dest.stat().st_size / 1e6:.1f} MB)")
        (OUT / "credits.json").write_text(json.dumps(credits, indent=1, ensure_ascii=False), encoding="utf-8")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

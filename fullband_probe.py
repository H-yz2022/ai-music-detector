r"""Step 2: do the faint generators (Mureka, Lyria) leave their trace outside the 5-16 kHz band?

Mureka and Lyria stay under 40% caught even when trained on (learning_test.py). Maybe their upsampling peaks sit in
another part of the spectrum. Songs (the first 60 s of each is streamed into memory; no audio is saved):
  mureka   50 random Mureka v9 songs (homura23/MUSIC8K, CC BY 4.0)
  lyria3   the 51 Lyria 3 songs (sleeping-ai/Lyria3)
  suno     20 Suno V5.5 songs (Kukedlc/suno-ai-music-dataset, CC BY 4.0): a generator with a known comb, as a check
  real     50 Jamendo songs (jamendolyrics/jam-alt, Creative Commons), never trained on
Every song: mono, its own sample rate (44.1 or 48 kHz), average dB spectrum (FFT 16384), then the fakeprint and the
precise comb in four bands: 0.5-2, 2-5, 5-16 (the band used now) and 16-20 kHz. Also the top of the content (MP3
files are often cut off around 16 kHz).
Per generator and band: the share of songs whose precise comb is above 95% of the real songs (5% expected by chance),
and the precise comb of the generator's mean fakeprint (peaks at the same place in every song add up there).
Read before running: a band "has a trace" for a generator if at least 30% of its songs beat the real songs' 95% mark
there, or its mean fakeprint has a comb at least 1.5x the real songs' mean fakeprint.

  python -I fullband_probe.py            # ~240 MB streamed, ~10 min; summary -> reports\fullband_probe.json
"""
import io
import json
import os
import random
import sys
import urllib.parse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import fakeprint as fpm  # noqa: E402
import fresh44_test as f44  # noqa: E402

HERE = Path(__file__).resolve().parent
SECONDS = 60.0
BANDS = [(500, 2000), (2000, 5000), (5000, 16000), (16000, 20000)]
f44.SECONDS = SECONDS


def jobs(seed=0):
    out = []
    for source, repo, folder, min_size, n in (("lyria3", "sleeping-ai/Lyria3", "", 0, 0),
                                              ("suno", "Kukedlc/suno-ai-music-dataset", "audio", 1_000_000, 20),
                                              ("real", "jamendolyrics/jam-alt", "subsets", 0, 50)):
        src = "real_jamendo" if source == "real" else source  # hf_jobs keeps only jam-alt's /audio/ files
        out += [(source, name.split("/", 1)[1], url) for _, name, (_, url) in f44.hf_jobs(src, repo, folder, min_size, n, seed)]
    from more44_test import hf_tree
    mu = sorted(x["path"] for x in hf_tree("homura23/MUSIC8K", "mureka_v9") if x["path"].lower().endswith(".mp3"))
    base = "https://huggingface.co/datasets/homura23/MUSIC8K/resolve/main/"
    out += [("mureka", Path(p).name, base + urllib.parse.quote(p)) for p in random.Random(seed).sample(mu, 50)]
    return out


def song(job):
    source, name, url = job
    blob, n_bytes = f44.mp3_start(url)
    with sf.SoundFile(io.BytesIO(blob)) as snd:
        sr = snd.samplerate
        x = snd.read(int(sr * SECONDS), dtype="float64", always_2d=True).mean(1)
    freqs, spec = fpm.average_spectrum_db(x[None, :], sr)
    top = float(freqs[np.flatnonzero(spec > spec.max() - 70)[-1]]) / 1000  # last bin within 70 dB of the peak
    fps, combs = [], []
    for lo, hi in BANDS:
        if hi > sr / 2:
            fps.append(None)
            combs.append((np.nan, np.nan))
            continue
        bf, fp = fpm.fakeprint(freqs, spec, lo, hi)
        fps.append((bf, fp.astype(np.float32)))
        combs.append(fpm.precise_comb(bf, fp))
    return source, name, sr, n_bytes, top, fps, combs


def main():
    todo = jobs()
    print(f"{len(todo)} songs", flush=True)
    rows, total = [], 0
    with ProcessPoolExecutor(4, initializer=f44.quiet_worker) as ex:
        futs = [ex.submit(song, j) for j in todo]
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as exc:
                print(f"  skipped a song ({exc})", flush=True)
                continue
            rows.append(r)
            total += r[3]
            if k % 20 == 0:
                print(f"  {k}/{len(todo)} songs, {total / 1e6:.0f} MB read", flush=True)
    print(f"Read {total / 1e6:.0f} MB in all; nothing saved")
    sources = ["real", "suno", "mureka", "lyria3"]
    res = {"read_mb": total / 1e6, "bands": [f"{lo / 1000:g}-{hi / 1000:g} kHz" for lo, hi in BANDS], "sources": {}}
    real_rows = [r for r in rows if r[0] == "real"]
    for s in sources:
        rs = [r for r in rows if r[0] == s]
        info = {"songs": len(rs), "sample_rates": sorted({r[2] for r in rs}),
                "top_khz_median": float(np.median([r[4] for r in rs])), "bands": {}}
        for b, (lo, hi) in enumerate(BANDS):
            ok = [r for r in rs if r[5][b] is not None]
            okr = [r for r in real_rows if r[5][b] is not None]
            if len(ok) < 5 or len(okr) < 5:
                continue
            sharp = np.array([r[6][b][0] for r in ok])
            cut = float(np.quantile([r[6][b][0] for r in okr], 0.95))
            # mean fakeprint over the songs at the most common sample rate (same frequency bins)
            sr0 = max({r[2] for r in ok}, key=lambda v: sum(r[2] == v for r in ok))
            same = [r for r in ok if r[2] == sr0]
            bf = same[0][5][b][0]
            m_sharp, m_spacing = fpm.precise_comb(bf, np.mean([r[5][b][1] for r in same], 0))
            info["bands"][res["bands"][b]] = {
                "beat_real_95": float(np.mean(sharp > cut)), "median_comb": float(np.median(sharp)),
                "mean_fp_comb": float(m_sharp), "mean_fp_spacing_hz": float(m_spacing), "songs": len(ok)}
        res["sources"][s] = info
    real_mean = {b: v["mean_fp_comb"] for b, v in res["sources"]["real"]["bands"].items()}
    print(f"\n{'':10s}" + "".join(f"{b:>30s}" for b in res["bands"]))
    print(f"{'':10s}" + "".join(f"{'beat real 95% | mean-fp comb':>30s}" for _ in res["bands"]))
    for s in sources:
        i = res["sources"][s]
        cells = []
        for b in res["bands"]:
            v = i["bands"].get(b)
            cells.append(f"{100 * v['beat_real_95']:9.0f}% | {v['mean_fp_comb']:5.1f} ({v['mean_fp_spacing_hz']:6.1f} Hz)"
                         if v else f"{'-':>30s}")
        print(f"{s:10s}" + "".join(f"{c:>30s}" for c in cells)
              + f"   top {i['top_khz_median']:.1f} kHz, {i['songs']} songs, {i['sample_rates']}")
    print("\nTrace (>= 30% beat the real 95% mark, or mean-fp comb >= 1.5x real's):")
    for s in ("suno", "mureka", "lyria3"):
        hits = [b for b, v in res["sources"][s]["bands"].items()
                if v["beat_real_95"] >= 0.30 or v["mean_fp_comb"] >= 1.5 * real_mean.get(b, np.inf)]
        print(f"  {s}: {', '.join(hits) if hits else 'none'}")
        res["sources"][s]["trace_bands"] = hits
    (HERE / "reports" / "fullband_probe.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

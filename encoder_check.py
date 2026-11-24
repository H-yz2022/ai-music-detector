r"""Control: is the big-data classifier spotting the MP3 encoder instead of AI?

Real and AI songs were MP3-encoded by different programs. This trains the
paper's classifier on all of features\*.npz, then scores the fresh songs twice:
as they are, and after running every fresh song (real and AI) through the exact
encoder the real training songs went through (fetch_data.to_match_format).
If AI songs are still caught after re-encoding, the model is not just
recognising SONICS' encoder. It also prints where the model's weights sit.

  python -I encoder_check.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import fakeprint as fpm  # noqa: E402
import fetch_data as fd  # noqa: E402
from audio_io import list_audio_files, load_audio, load_audio_bytes  # noqa: E402
from detect import make_model  # noqa: E402

X, gens = [], []
for p in sorted((PROJ / "features").glob("*.npz")):
    d = np.load(p)
    X.append(d["fp"].astype(np.float32))
    gens += list(d["gens"])
X, gens = np.concatenate(X), np.array(gens)
model = make_model().fit(X, gens != "real")
coef = model.coef_[0]
f = d["freqs"]
for lo, hi in ((1000, 3000), (3000, 5000), (5000, 7000), (7000, 7500), (7500, 8000)):
    sel = (f >= lo) & (f < hi)
    print(f"|weight| share {lo / 1000:g}-{hi / 1000:g} kHz: {np.abs(coef[sel]).sum() / np.abs(coef).sum():.0%} "
          f"of weight, {sel.mean():.0%} of bins")

rows = {"as is": [], "re-encoded": []}
tg = []
for folder in (PROJ / "data_fresh" / "real", PROJ / "data_fresh" / "ai"):
    for path in list_audio_files(folder):
        name = path.name.lower()
        tg.append("suno" if "_suno_" in name else "udio" if "_udio_" in name else "real")
        audio, sr = load_audio(path, target_sr=16000, max_duration=30)
        rows["as is"].append(fpm.fakeprint(*fpm.average_spectrum_db(audio, sr), 1000, 8000)[1])
        data, src_sr = sf.read(str(path), dtype="float64", always_2d=True)
        audio, sr = load_audio_bytes(fd.to_match_format(data, src_sr), target_sr=16000, max_duration=30)
        rows["re-encoded"].append(fpm.fakeprint(*fpm.average_spectrum_db(audio, sr), 1000, 8000)[1])
tg = np.array(tg)
print(f"\n{'fresh songs':12s}{'real called real':>18s}{'Suno caught':>13s}{'Udio caught':>13s}{'AUC all AI':>12s}")
for k, fps in rows.items():
    p = model.predict_proba(np.stack(fps).astype(np.float32))[:, 1]
    print(f"{k:12s}{100 * np.mean(p[tg == 'real'] < 0.5):17.0f}%{100 * np.mean(p[tg == 'suno'] >= 0.5):12.0f}%"
          f"{100 * np.mean(p[tg == 'udio'] >= 0.5):12.0f}%{roc_auc_score(tg != 'real', p):12.3f}")

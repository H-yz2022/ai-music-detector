r"""Demo: score one song and show why (for the Colab notebook demo.ipynb, or locally).

  from demo import show
  show("song.mp3")                  # rank among real songs + plot of the fakeprint
  show("song.mp3", base_rate=0.1)   # also a rough chance of AI if 10% of songs are AI
  show("song.mp3", joint=True)      # classifier or precise comb (better for new generators with a comb)
  show("song.mp3", mode="whole")    # the whole song instead of 30 s parts (more AI caught, more false alarms)

The models are stored as plain numbers (demo_models\: weights, cut-offs, real-song scores; no pickles), so any
scikit-learn version works. Rebuild them after retraining:
  python -I demo.py                 # demo_models\ from model_twoband.joblib + model_family.joblib, and the zip in dist\
"""
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
MODELS = HERE / "demo_models"
CODE = ["detect.py", "fakeprint.py", "audio_io.py", "fetch_data.py", "synth.py", "report.py", "demo.py"]


class LinearModel:
    """A fitted logistic regression as plain numbers: predict_proba and classes_, like scikit-learn's."""

    def __init__(self, coef, intercept, classes):
        self.coef_, self.intercept_, self.classes_ = coef, intercept, np.array(classes)

    def predict_proba(self, X):
        z = np.asarray(X, np.float64) @ self.coef_.T + self.intercept_
        if self.coef_.shape[0] == 1:  # yes / no
            p = 1 / (1 + np.exp(-z[:, 0]))
            return np.column_stack([1 - p, p])
        z -= z.max(1, keepdims=True)  # one class per family (multinomial)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)


def _flatten(obj, prefix, arrays, meta):
    """Nested dicts of arrays and numbers -> {path: array} + JSON-able structure."""
    if isinstance(obj, dict):
        return {k: _flatten(v, f"{prefix}/{k}", arrays, meta) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return {"__list__": [_flatten(v, f"{prefix}/{i}", arrays, meta) for i, v in enumerate(obj)]}
    if isinstance(obj, np.ndarray):
        arrays[prefix] = obj
        return {"__array__": prefix}
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def _unflatten(node, arrays):
    if isinstance(node, dict):
        if "__array__" in node:
            return arrays[node["__array__"]]
        if "__list__" in node:
            return [_unflatten(v, arrays) for v in node["__list__"]]
        return {k: _unflatten(v, arrays) for k, v in node.items()}
    return node


def export(out=MODELS):
    """demo_models/{twoband,family}.{npz,json} from the trained models."""
    import joblib
    out.mkdir(exist_ok=True)
    for name, src in (("twoband", "model_twoband.joblib"), ("family", "model_family.joblib")):
        b = dict(joblib.load(HERE / src))
        m = b.pop("model")
        b["model"] = {"coef": m.coef_, "intercept": m.intercept_, "classes": [str(c) for c in m.classes_]}
        b.pop("trained_on", None)
        arrays = {}
        meta = _flatten(b, name, arrays, None)
        np.savez_compressed(out / f"{name}.npz", **arrays)
        (out / f"{name}.json").write_text(json.dumps(meta, indent=1))
        X = np.random.default_rng(0).random((5, m.coef_.shape[1]))
        lm = load_bundle(name, out)["model"]
        err = np.abs(lm.predict_proba(X) - m.predict_proba(X)).max()
        print(f"  {name}: {m.coef_.shape[1]} features, {len(m.classes_)} classes; same probabilities (max diff {err:.1e})")


def load_bundle(name, folder=MODELS):
    arrays = dict(np.load(folder / f"{name}.npz"))
    b = _unflatten(json.loads((folder / f"{name}.json").read_text()), arrays)
    m = b["model"]
    b["model"] = LinearModel(m["coef"], m["intercept"], m["classes"])
    return b


def make_zip(out=HERE / "dist" / "ai-music-detector-demo.zip"):
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in CODE:
            z.write(HERE / f, f"ai-music-detector/{f}")
        for f in sorted(MODELS.iterdir()):
            z.write(f, f"ai-music-detector/demo_models/{f.name}")
    print(f"  {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def score(path, base_rate=0.0, joint=False, mode="windows"):
    """(text line as predict prints it, plot data or None). Plot data only for 44.1 kHz files (the two-band model)."""
    import detect
    two, one = load_bundle("twoband"), load_bundle("family")
    import soundfile as sf
    sr = sf.info(str(path)).samplerate
    if sr < two["min_sr"]:  # lower rates: the one-band 16 kHz model, after the same MP3 conversion as its tests
        args = SimpleNamespace(sr=16000, fmin=1000.0, fmax=8000.0, max_duration=one.get("max_duration", 30.0),
                               match_format=True)
        return detect.predict_one_band(path, one, args) + "  (16 kHz model: AI / real only)", None
    data, sr = detect.read_start(path, detect.read_seconds(two, mode))
    line = detect.predict_two_band(path, two, data, sr, joint=joint, mode=mode, base_rate=base_rate)
    r = detect.two_band_scores(data, sr, two, mode)  # the same numbers predict used
    cal, x = r["cal"], r["x"]
    ref, val = (cal["real_joint"], float(detect.joint_score(cal["real_scores"], r["p"], r["s16"], r["s44"]))) \
        if joint else (cal["real_scores"]["clf"], r["p"])
    rank = float(np.searchsorted(ref, val, side="right") / len(ref))
    n0 = len(two["bands"][0]["freqs"])
    bands = [(two["bands"][0]["freqs"], x[:n0], (r["s16"], r["e16"]), r["c16"]),
             (two["bands"][1]["freqs"], x[n0:], (r["s44"], r["e44"]), r["c44"])]
    return line, {"rank": rank, "bands": bands, "seconds": r["seconds"], "how": r["how"]}


def analyze(path, two=None, one=None, mode="windows"):
    """Everything the web app (app.py) shows for one song, as plain values:
    level ("likely AI" / "unsure" / "likely real"), rank (share of real songs below; None for the 16 kHz model),
    p (the model's P(AI)), cutoff_p, seconds and how (e.g. "median of 8 windows of 30 s"), and bands: a list of
    (name, freqs in Hz, fakeprint, comb spacing in Hz or 0 if no exact comb)."""
    import soundfile as sf
    import detect
    import fakeprint as fpm
    two = two or load_bundle("twoband")
    one = one or load_bundle("family")
    sr = sf.info(str(path)).samplerate
    if sr < two["min_sr"]:  # the 16 kHz model, after the same 16 kHz, 40 kbps MP3 conversion as its tests
        from audio_io import load_audio_bytes
        from fetch_data import to_match_format
        dur = one.get("max_duration", 30.0)
        data, src_sr = detect.read_start(path, dur)
        audio, a_sr = load_audio_bytes(to_match_format(data, src_sr), target_sr=one["sr"], max_duration=dur)
        f, fp = fpm.fakeprint(*fpm.average_spectrum_db(audio, a_sr), one["fmin"], one["fmax"])
        sharp, spacing = fpm.precise_comb(f, fp)
        p = float(detect.ai_proba(one["model"], fp[None, :])[0])
        return {"model": "16 kHz", "level": "likely AI" if p > one["cutoff"] else "likely real", "rank": None,
                "p": p, "cutoff_p": one["cutoff"], "seconds": min(len(data) / src_sr, dur), "how": "first 30 s",
                "sample_rate": sr,
                "bands": [("1–8 kHz", f, fp, spacing if sharp > detect.SHARPNESS_THRESHOLD else 0.0)]}
    data, sr = detect.read_start(path, detect.read_seconds(two, mode))
    r = detect.two_band_scores(data, sr, two, mode)  # the same numbers detect.py predict uses
    cal, p, x = r["cal"], r["p"], r["x"]
    rank = float(np.searchsorted(cal["real_scores"]["clf"], p, side="right") / len(cal["real_scores"]["clf"]))
    n0 = len(two["bands"][0]["freqs"])
    bands = [("5–16 kHz", two["bands"][1]["freqs"], x[n0:], r["e44"] if r["s44"] > r["c44"] else 0.0),
             ("1–8 kHz", two["bands"][0]["freqs"], x[:n0], r["e16"] if r["s16"] > r["c16"] else 0.0)]
    level = "likely AI" if rank > detect.LEVELS[1] else "unsure" if rank > detect.LEVELS[0] else "likely real"
    return {"model": "44.1 kHz", "level": level, "rank": rank, "p": p, "cutoff_p": cal["cutoff"],
            "seconds": r["seconds"], "how": r["how"], "sample_rate": sr,
            "n_real": len(cal["real_scores"]["clf"]), "bands": bands}


def figure(name, line, plot):
    """Plotly figure: the rank among real songs, then the fakeprint of each band (comb lines if there is a comb)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    lo, hi = 90, 99
    r = 100 * plot["rank"]
    level = "likely AI" if r > hi else "unsure" if r > lo else "likely real"
    color = {"likely AI": "#c0392b", "unsure": "#d68910", "likely real": "#1e8449"}[level]
    fig = make_subplots(rows=3, cols=1, row_heights=[0.16, 0.42, 0.42], vertical_spacing=0.11,
                        subplot_titles=[f"Rank among real songs: higher than {r:.1f}%  ->  {level}",
                                        "Fakeprint, 1–8 kHz (16 kHz audio)", "Fakeprint, 5–16 kHz (44.1 kHz audio)"])
    for a, b, c, label in ((0, lo, "#d5f5e3", "likely real"), (lo, hi, "#fdebd0", "unsure"),
                           (hi, 100, "#fadbd8", "likely AI")):
        fig.add_trace(go.Bar(x=[b - a], y=[""], base=[a], orientation="h", marker_color=c, name=label,
                             hoverinfo="name", showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=[r], y=[""], mode="markers", marker=dict(size=16, color=color, symbol="diamond"),
                             name="this song", hovertemplate="%{x:.1f}%<extra>this song</extra>", showlegend=False), 1, 1)
    fig.update_xaxes(range=[0, 100], ticksuffix="%", row=1, col=1)
    for k, (f, fp, (sharp, spacing), cut) in enumerate(plot["bands"], start=2):
        fig.add_trace(go.Scatter(x=f / 1000, y=fp, mode="lines", line=dict(width=1, color="#2e4053"),
                                 showlegend=False, hovertemplate="%{x:.3f} kHz: %{y:.2f}<extra></extra>"), k, 1)
        if sharp > cut and spacing > 0:  # exact comb: mark its repeats
            for h in np.arange(np.ceil(f[0] / spacing), np.floor(f[-1] / spacing) + 1) * spacing:
                fig.add_vline(x=h / 1000, line=dict(color="rgba(192,57,43,0.25)", width=1), row=k, col=1)
            fig.layout.annotations[k - 1].text += f"  — peaks repeat every {spacing:.2f} Hz"
        fig.update_xaxes(title_text="kHz", row=k, col=1)
    how = plot["how"]
    fig.update_layout(height=640, title=f"{name}: {how}", margin=dict(l=40, r=20, t=80, b=40), bargap=0.3)
    return fig


def show(path, base_rate=0.0, joint=False, mode="windows"):
    """Print predict's answer for one song and, for 44.1 kHz files, show the plot. mode "windows" (default: median
    over 30 s parts) or "whole" (the whole song, own cut-off; flagged 7% of new real songs, not 5%)."""
    line, plot = score(path, base_rate, joint, mode)
    print(line)
    if plot is not None:
        figure(Path(path).name, line, plot).show()


if __name__ == "__main__":
    print("Exporting the models as plain numbers ...")
    export()
    make_zip()

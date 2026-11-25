"""AI music detector: fakeprints + logistic regression (after Afchar et al., ISMIR 2025).

  python detect.py selftest                          # no data needed
  python detect.py train --real data/real --ai data/ai
  python detect.py predict song1.mp3 song2.wav       # 44.1 kHz files: model_twoband.joblib (both bands, one
                                                     # boundary per generator family, or the spacing bank);
                                                     # lower rates: model_family.joblib (1-8 kHz)
  python detect.py predict song.mp3 --match-format   # low-rate files: first the same 16 kHz, 40 kbps MP3 as our tests
  python detect.py predict song.mp3 --joint          # 44.1 kHz files: classifier or comb (for new generators)
  python detect.py predict song.mp3 --no-bank        # 44.1 kHz files: the classifier alone (better after speed changes)
  python detect.py predict song.mp3 --model model_twoband_v1.joblib   # the model before Nov 25, 2026 (one boundary)
  python detect.py predict song.mp3 --base-rate 0.1  # also a rough chance of AI, if 10% of songs are AI
  44.1 kHz files: the median over 30 s windows (up to 8); --whole: one fakeprint of the whole song (up to 8 min,
  own cut-off; catches more AI but flagged 7% of new real songs, not 5%: wholecheck_test.py);
  --first-30s scores only the start.
  44.1 kHz files: each song is ranked among real songs (likely real / unsure / likely AI) instead of AI / real.
  Prints the exact comb spacing when there is one.
"""
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Find the sibling modules even when run as "python -I detect.py" (isolated mode,
# recommended when reading downloaded audio).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fakeprint as fpm
import synth
from audio_io import list_audio_files, load_audio
from report import write_report

HERE = Path(__file__).resolve().parent
COMB_THRESHOLD = 0.2  # comb score above this = "regular peaks found"
# Precise comb (fakeprint.precise_comb, top 5 repeat distances): this cut-off flags 5% of 7,894
# real FMA clips in the setup below (16 kHz, 1-8 kHz band, first 30 s, 40 kbps MP3). See unseen_test.py.
SHARPNESS_THRESHOLD = 9.23
CALIBRATED = (16000, 1000.0, 8000.0, 30.0)
# Exact comb spacings measured in our tests (unseen_test.py), to say what a comb looks like.
KNOWN_SPACINGS = [(21.533, "DiffRhythm / Riffusion: 44,100 / 2,048"), (50.0, "MusicGen / SongGen"),
                  (86.133, "ACE-Step: 44,100 / 512"), (100.0, "AudioLDM"),
                  (172.266, "Stable Audio 2.0: 44,100 / 256"), (200.0, "Suno / Brev / ElevenLabs"),
                  (689.0625, "MiniMax Music 3.0: 44,100 / 64"),  # fresh44_test.py
                  (125.0, "HeartMuLa"),  # more44_test.py
                  (25.0, "older Suno"), (75.0, "older Suno")]
PAPER = "<a href='https://arxiv.org/abs/2506.19108'>Afchar et al., ISMIR 2025</a>"


def make_model():
    # Same as the authors' train_test_regressor.py (they use sklearn's default
    # max_iter; a higher cap just avoids convergence warnings).
    return LogisticRegression(class_weight="balanced", max_iter=5000)


def cv_probabilities(X, y, seed=0):
    """P(AI) for each clip from a model that never saw that clip (k-fold CV)."""
    k = int(min(5, np.bincount(y).min()))
    if k < 2:
        return None
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return cross_val_predict(make_model(), X, y, cv=cv, method="predict_proba")[:, 1]


def pct(values):
    values = list(values)
    return f"{100 * np.mean(values):.0f}%" if values else "–"


def cmd_selftest(args):
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plan = [
        ("real", lambda: synth.make_real_clip(rng, dur=args.duration), args.n),
        ("ai", lambda: synth.make_ai_clip(rng, synth.DECODERS["A"], dur=args.duration), args.n),
        ("ai-shifted", lambda: synth.speed_up(synth.make_ai_clip(rng, synth.DECODERS["A"], dur=args.duration)),
         max(4, args.n // 2)),
        ("ai-unseen", lambda: synth.make_ai_clip(rng, synth.DECODERS["B"], dur=args.duration), max(4, args.n // 2)),
    ]
    items = []
    t0 = time.time()
    for group, make, count in plan:
        for i in range(count):
            audio = make()
            if i == 0:  # keep one example per group to listen to
                wavfile.write(out.parent / f"example_{group}.wav", synth.SR, (audio * 32767).astype(np.int16))
            items.append({"name": f"{group}-{i + 1:02d}", "group": group, **fpm.extract(audio[None, :], synth.SR)})
        print(f"  made {count} '{group}' clips ({time.time() - t0:.0f}s)")

    train = [it for it in items if it["group"] in ("real", "ai")]
    unseen = [it for it in items if it["group"] not in ("real", "ai")]  # never trained on
    X = np.stack([it["fp"] for it in train])
    y = np.array([it["group"] == "ai" for it in train], dtype=int)
    for it, p in zip(train, cv_probabilities(X, y)):
        it["prob"] = float(p)
    model = make_model().fit(X, y)
    for it, p in zip(unseen, model.predict_proba(np.stack([it["fp"] for it in unseen]))[:, 1]):
        it["prob"] = float(p)

    def acc(group, want_ai):
        return pct((it["prob"] >= 0.5) == want_ai for it in items if it["group"] == group)

    def comb_rate(group):
        return pct(it["comb_score"] >= COMB_THRESHOLD for it in items if it["group"] == group)

    def spacing(group):
        return np.median([it["spacing_hz"] for it in items if it["group"] == group])

    cards = [
        ("Real clips called real", acc("real", False), "logistic regression, 5-fold CV"),
        ("AI clips caught", acc("ai", True), "decoder A, 5-fold CV"),
        ("Pitch-shifted AI caught", acc("ai-shifted", True), "decoder A, +1 semitone, never trained on"),
        ("Unseen decoder caught", acc("ai-unseen", True), "decoder B, never trained on"),
        ("Comb found: AI / shifted / unseen / real",
         " / ".join(comb_rate(g) for g in ("ai", "ai-shifted", "ai-unseen", "real")),
         f"comb score ≥ {COMB_THRESHOLD}, no training at all"),
        ("Peak spacing: A / shifted / B",
         f"{spacing('ai'):.1f} / {spacing('ai-shifted'):.1f} / {spacing('ai-unseen'):.1f} Hz",
         f"expected {44100 / 320:.1f} / {44100 / 320 * 18 / 17:.1f} / {44100 / 512:.1f} Hz"),
    ]
    intro = (f"A check of the method in {PAPER}, run on synthetic audio. "
             "<b>Real</b> clips are synthesised notes and noise bursts. <b>AI</b> clips come from a random, "
             "untrained transposed-convolution decoder with strides (8, 5, 4, 2), like Encodec. "
             "<b>Pitch-shifted</b> clips are the same kind of AI audio played back about 1 semitone higher, "
             "a change the paper leaves to future work. <b>Unseen</b> clips use a different decoder "
             "(4, 4, 4, 4, 2). The classifier is trained on Real vs AI only. "
             "If the paper is right, the decoder alone leaves evenly spaced peaks, at multiples of its latent frame rate.")
    notes = ["This tests the <b>mechanism</b>, not real-world accuracy: the synthetic AI clips don't sound like music. "
             "Run <code>python detect.py train</code> on real songs and Suno/Udio songs for that.",
             "Listen to <code>example_real.wav</code> and <code>example_ai.wav</code> next to this report."]
    path = write_report(out, "AI Music Detector · Self-test", intro, items, cards, COMB_THRESHOLD,
                        coef=model.coef_[0], notes=notes)
    print("\n".join(f"  {l}: {v}  ({s})" for l, v, s in cards))
    print(f"Report: {path}")


def featurize_folder(folder, group, sr, fmin, fmax, max_duration=180.0):
    items = []
    for path in list_audio_files(folder):
        try:
            audio, sr = load_audio(path, target_sr=sr, max_duration=max_duration)
        except Exception as exc:  # unreadable file: skip it, keep going
            print(f"  skip {path.name}: {exc}")
            continue
        items.append({"name": path.name, "group": group, **fpm.extract(audio, sr, fmin, fmax)})
        print(f"  {group:4s} {path.name}  comb={items[-1]['comb_score']:.2f}")
    return items


def cmd_train(args):
    for folder in (args.real, args.ai):
        if not Path(folder).is_dir():
            sys.exit(f"Folder not found: {folder}")
    band = (args.sr, args.fmin, args.fmax, args.max_duration)
    items = featurize_folder(args.real, "real", *band) + featurize_folder(args.ai, "ai", *band)
    y = np.array([it["group"] == "ai" for it in items], dtype=int)
    if y.sum() == 0 or y.sum() == len(y):
        sys.exit("Need at least one readable file in BOTH the real and the AI folder.")
    X = np.stack([it["fp"] for it in items])

    probs = cv_probabilities(X, y)
    if probs is None:
        print("  (fewer than 2 files in a class: skipping cross-validation)")
    else:
        for it, p in zip(items, probs):
            it["prob"] = float(p)
    model = make_model().fit(X, y)
    joblib.dump({"model": model, "sr": args.sr, "fmin": args.fmin, "fmax": args.fmax,
                 "n_features": X.shape[1]}, args.model)

    def acc(group, want_ai):
        return pct((it["prob"] >= 0.5) == want_ai for it in items if it["group"] == group and "prob" in it)

    def comb_rate(group):
        return pct(it["comb_score"] >= COMB_THRESHOLD for it in items if it["group"] == group)

    n_real, n_ai = int((y == 0).sum()), int(y.sum())
    cards = [
        ("Files", f"{n_real} real · {n_ai} AI",
         f"at {args.sr / 1000:g} kHz, band {args.fmin / 1000:g}–{args.fmax / 1000:g} kHz, "
         f"first {args.max_duration:g} s"),
        ("Real called real", acc("real", False), "cross-validated"),
        ("AI caught", acc("ai", True), "cross-validated"),
        ("Comb found: AI / real", f"{comb_rate('ai')} / {comb_rate('real')}", f"comb score ≥ {COMB_THRESHOLD}"),
    ]
    intro = (f"Fakeprint + logistic regression ({PAPER}) on your own files. Each P(AI) comes from a model "
             "that did not see that file during training (k-fold cross-validation).")
    notes = ["Keep both groups in the <b>same format</b> (e.g. all MP3 at the same bitrate). Otherwise the model can "
             "learn 'MP3 vs WAV' instead of 'AI vs real'.",
             "With only a few songs per group, treat the accuracy as a rough first look, not a final number."]
    path = write_report(args.out, "AI Music Detector · Your data", intro, items, cards, COMB_THRESHOLD,
                        coef=model.coef_[0], notes=notes)
    print("\n".join(f"  {l}: {v}" for l, v, _ in cards))
    print(f"Model:  {args.model}\nReport: {path}")


def known_spacing(hz):
    """' (like X)' if an exact spacing (or a small multiple of it) matches one seen in our tests."""
    for k in (1, 2, 3, 4):  # direct matches first, then multiples
        for known, who in KNOWN_SPACINGS:
            if abs(hz - k * known) < 0.02 * k:
                return f" (like {who}{'' if k == 1 else f', x{k}'})"
    return ""


def read_start(path, max_duration):
    """(samples x channels array, sample rate) of the first max_duration seconds. Needs soundfile."""
    import soundfile as sf  # optional dependency, only needed here
    with sf.SoundFile(str(path)) as snd:
        sr = snd.samplerate
        data = snd.read(int(sr * max_duration) if max_duration else -1, dtype="float64", always_2d=True)
    return data, sr


def formatted_spectrum(data, src_sr, sr, mp3_level, max_duration):
    """Average dB spectrum after the same MP3 conversion as the training songs (same steps as
    build_fakeprints.featurize): (freqs, spectrum)."""
    from audio_io import load_audio_bytes
    from fetch_data import to_format
    audio, sr = load_audio_bytes(to_format(data, src_sr, sr, mp3_level), target_sr=sr, max_duration=max_duration)
    return fpm.average_spectrum_db(audio, sr)


def content_top_khz(freqs, spec, drop_db=30.0):
    """Highest 500 Hz band within drop_db of the 5-8 kHz level (as hq_test.content_top), in kHz."""
    lo = np.arange(0, freqs[-1] + 1, 500.0)[:-1]  # band starts, up to Nyquist (freqs[-1])
    coarse = np.array([spec[(freqs >= a) & (freqs < a + 500)].mean() for a in lo])
    ok = np.flatnonzero(coarse > coarse[(lo >= 5000) & (lo < 8000)].mean() - drop_db)
    return (lo[ok[-1]] + 500) / 1000 if len(ok) else 0.0


def ai_proba(model, X):
    """P(AI) per row. Yes/no models give it directly; one-class-per-family models (family_test.py,
    train_twoband.py --family) give 1 - P(real)."""
    proba, classes = model.predict_proba(X), [str(c) for c in model.classes_]
    return 1 - proba[:, classes.index("real")] if "real" in classes else proba[:, 1]


def most_like(model, x):
    """', most like <family>' for a one-class-per-family model, else ''."""
    classes = [str(c) for c in model.classes_]
    if "real" not in classes:
        return ""
    proba = model.predict_proba(x[None, :])[0]
    return f", most like {max((q, c) for q, c in zip(proba, classes) if c != 'real')[1]}"


def joint_score(refs, clf, s16, s44):
    """Joint rule (twoband_test.py, train_twoband.py): the highest of the three scores (classifier P(AI), precise
    comb at 1-8 kHz, at 5-16 kHz), each as the share of real training songs it is above (0.97 = above 97%)."""
    pct = lambda ref, x: np.searchsorted(ref, x, side="right") / len(ref)  # noqa: E731
    return np.maximum.reduce([pct(refs["clf"], clf), pct(refs["s16"], s16), pct(refs["s44"], s44)])


LEVELS = (0.90, 0.99)  # below: likely real; between: unsure; above: likely AI (share of real songs below the score)


def rate_text(score, real_sorted, ai_sorted=None, base_rate=0.0):
    """How unusual a score is for a real song, instead of a plain AI / real answer.
    real_sorted: cross-validated scores of real training songs (sorted). Returns (level, text).
    The share of real songs below the score holds on real songs never trained on (newreal_test.py: the 90 / 95 / 99%
    levels flag 11 / 5.6 / 0.9% of 1,077 new real songs). With base_rate (the share of AI songs where the detector
    is used) and ai_sorted (cross-validated AI scores), also a rough chance that the song is AI, from how much more
    often AI songs than real songs land in the same score range (u = -log10(share of real songs above): bins 0-0.5,
    0.5-1, ..., 3-4; real songs fill bin a-b with share 10^-a - 10^-b by construction). Rough, because new
    generators score lower than the training ones."""
    n = len(real_sorted)
    q = float(np.searchsorted(real_sorted, score, side="right") / n)
    above = max(1.0 - q, 1.0 / n)
    level = "likely AI" if q > LEVELS[1] else "unsure" if q > LEVELS[0] else "likely real"
    txt = f"higher than {100 * q:.1f}% of real songs"
    if q >= LEVELS[0]:
        txt += f" (about 1 in {round(1 / above):,} real songs scores this high)"
    if base_rate and ai_sorted is not None:
        edges = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, np.inf])
        to_u = lambda qq: -np.log10(np.maximum(1.0 - qq, 1.0 / n))  # noqa: E731
        u_ai = to_u(np.searchsorted(real_sorted, ai_sorted, side="right") / n)
        k = int(np.searchsorted(edges, to_u(q), side="right") - 1)
        ai_share = max(float(np.mean((u_ai >= edges[k]) & (u_ai < edges[k + 1]))), 0.5 / len(ai_sorted))
        real_share = 10.0 ** -edges[k] - (10.0 ** -edges[k + 1] if np.isfinite(edges[k + 1]) else 0.0)
        ratio = ai_share / real_share
        chance = base_rate * ratio / (base_rate * ratio + 1 - base_rate)
        c = "<1" if chance < 0.01 else ">99" if chance > 0.99 else f"~{100 * chance:.0f}"
        txt += f"; if {100 * base_rate:g}% of songs are AI: {c}% chance it is AI (rough)"
    return level, txt


def comb_note(sharp, exact, cut):
    return f" -> peaks repeat every {exact:.3f} Hz{known_spacing(exact)}" if sharp > cut else " (no exact comb)"


MAX_WINDOWS = 8  # window_test.py: up to 8 windows of 30 s (the first 4 minutes)


def two_band_window(data, src_sr, bundle):
    """Both fakeprints of one window: (features, (s16, exact16), (s44, exact44), content top in kHz)."""
    fps, combs = [], []
    for band in bundle["bands"]:
        freqs, spec = formatted_spectrum(data, src_sr, band["sr"], band["mp3_level"], bundle["max_duration"])
        band_f, fp = fpm.fakeprint(freqs, spec, band["fmin"], band["fmax"])
        if len(band_f) != len(band["freqs"]):
            raise ValueError(f"{len(band_f)} frequency bins, but the model expects {len(band['freqs'])}")
        fps.append(fp)
        combs.append(fpm.precise_comb(band_f, fp))
    return np.concatenate(fps), combs[0], combs[1], content_top_khz(freqs, spec)  # top: of the 44.1 kHz band


def use_whole(bundle, n_samples, src_sr, mode):
    """Whole-song scoring (--whole): only if the model has a whole-song calibration (calibrate_whole.py) and the
    song is long enough (60 s)."""
    w = bundle.get("whole")
    return mode == "whole" and w is not None and n_samples >= w["min_seconds"] * src_sr


def read_seconds(bundle, mode):
    """How many seconds of the song a mode needs."""
    seconds = bundle["max_duration"] * (1 if mode == "first" else MAX_WINDOWS)
    if mode == "whole" and "whole" in bundle:
        seconds = max(seconds, bundle["whole"]["max_seconds"])
    return seconds


def two_band_scores(data, src_sr, bundle, mode="windows", bank=False):
    """The numbers behind one two-band answer (shared by predict, demo.py and app.py): a dict with
    cal (where the cut-offs and real-song scores come from), p, s16 / e16, s44 / e44 (precise comb score and exact
    spacing per band), c16 / c44 (comb cut-offs), x (the fakeprint shown and used for "most like"), top (content
    top in kHz), seconds (audio used) and how (e.g. "median of 8 windows"). With `bank` (windows only, if the model
    has a spacing bank): bank_p (the bank score as the share of real songs below, median over the windows) and
    bank_i (the spacing that matched best)."""
    if use_whole(bundle, len(data), src_sr, mode):
        cal = bundle["whole"]
        seg = data[: int(cal["max_seconds"] * src_sr)]
        x, (s16, e16), (s44, e44), top = two_band_window(seg, src_sr, dict(bundle, max_duration=cal["max_seconds"]))
        p = float(ai_proba(bundle["model"], x[None, :])[0])
        c16, c44 = (float(np.quantile(cal["real_scores"][k], 0.95)) for k in ("s16", "s44"))
        return dict(cal=cal, p=p, s16=s16, e16=e16, s44=s44, e44=e44, c16=c16, c44=c44, x=x, top=top,
                    seconds=len(seg) / src_sr, how=f"whole song ({len(seg) / src_sr:.0f} s)")
    w = int(src_sr * bundle["max_duration"])
    n = min(MAX_WINDOWS, len(data) // w) if mode != "first" else 1
    n = n if n >= 2 else 1
    res = [two_band_window(data[k * w:(k + 1) * w], src_sr, bundle) for k in range(n)]
    ps = ai_proba(bundle["model"], np.array([r[0] for r in res]))
    c16, c44 = bundle["comb_cutoffs"]
    # The exact spacing: from the window and band where the comb stands out most (score / that band's cut-off)
    best = max(range(n), key=lambda k: max(res[k][1][0] / c16, res[k][2][0] / c44))
    seconds = min(len(data), n * w) / src_sr
    extra = {}
    if bank and "bank" in bundle:
        n0 = len(bundle["bands"][0]["freqs"])
        shares, idx = fpm.bank_score(np.asarray(bundle["bands"][1]["freqs"], dtype=np.float64),
                                     np.array([r[0][n0:] for r in res]), bundle["bank"])
        extra = {"bank_p": float(np.median(shares)), "bank_i": int(idx[int(np.argmax(shares))])}
    return dict(cal=bundle, p=float(np.median(ps)), **extra,
                s16=float(np.median([r[1][0] for r in res])), e16=res[best][1][1],
                s44=float(np.median([r[2][0] for r in res])), e44=res[best][2][1], c16=c16, c44=c44,
                x=res[int(np.argsort(ps)[len(ps) // 2])][0],  # the median window
                top=float(np.median([r[3] for r in res])), seconds=seconds,
                how=f"median of {n} windows of {bundle['max_duration']:g} s" if n > 1 else
                f"first {seconds:.0f} s")


def predict_two_band(f, bundle, data, src_sr, joint=False, mode="windows", base_rate=0.0, bank=False):
    """Two fakeprints (1-8 kHz of 16 kHz audio, 5-16 kHz of 44.1 kHz audio) side by side (train_twoband.py).
    mode "windows" (default): the song is cut into 30 s windows (up to 8) and the median of each score counts
    (window_test.py; on 1,200 new songs, wholecheck_test.py: 3.3% of real songs flagged, 95.8% of AI caught).
    mode "whole" (--whole; songs of 60 s or more, if the model has a whole-song calibration): one fakeprint of the
    whole song (up to 8 min) with its own cut-off. It catches more AI (99.7%), but flagged 7.0% of new real songs,
    not the 5% it was calibrated for, so it is not the default. mode "first": the first 30 s only.
    With `joint`, the verdict is classifier or precise comb, with one joint cut-off.
    With `bank` (predict's default since Nov 25, 2026; windows only): classifier or spacing bank, the comb checked
    only at the exact spacings of known generators (known_spacing_test.py, family_bank_test.py: with the per-family
    model, 83.4% vs 78.3% of fresh44's AI songs and 5.2% vs 6.3% of never-used real songs flagged, compared with the
    one-boundary classifier alone; but a speed change moves the spacing, and generators with no comb such as Lyria
    are caught less often).
    If the model has the real songs' scores (train_twoband.py), the answer is how the song ranks among real songs
    and a level (likely real / unsure / likely AI) instead of AI / real (rate_text)."""
    r = two_band_scores(data, src_sr, bundle, mode, bank=bank)
    cal, p, x, top = r["cal"], r["p"], r["x"], r["top"]
    s16, e16, s44, e44, c16, c44 = (r[k] for k in ("s16", "e16", "s44", "e44", "c16", "c44"))
    how = f", {r['how']}"
    sharp, exact, cut = (s16, e16, c16) if s16 / c16 >= s44 / c44 else (s44, e44, c44)
    is_ai = p > cal["cutoff"]
    verdict = "AI" if is_ai else "real"
    rates = "ai_scores" in cal  # models saved before Nov 23, 2026 only have the 5% cut-off
    if rates and not joint:
        level, verdict = rate_text(p, cal["real_scores"]["clf"], cal["ai_scores"]["clf"], base_rate)
        verdict = f"{level}: {verdict}"
        is_ai = level != "likely real"
    if joint:
        js = float(joint_score(cal["real_scores"], p, s16, s44))
        by_comb = js > cal["joint_cutoff"] and not is_ai  # the classifier alone would say real
        is_ai = js > cal["joint_cutoff"]
        verdict = f"joint={js:.3f} -> {'AI' if is_ai else 'real'}{' (by comb)' if by_comb else ''}"
        if rates:
            level, txt = rate_text(js, cal["real_joint"], cal["ai_scores"]["joint"], base_rate)
            by_comb = level != "likely real" and p <= np.quantile(cal["real_scores"]["clf"], LEVELS[0])
            verdict = f"joint={js:.3f} -> {level}{' (by comb)' if by_comb else ''}: {txt}"
            is_ai = level != "likely real"
    if bank and "bank_p" in r:
        bk = cal["bank"]
        clf_p = float(np.searchsorted(cal["real_scores"]["clf"], p, side="right") / len(cal["real_scores"]["clf"]))
        bs = max(clf_p, r["bank_p"])
        level, txt = rate_text(bs, bk["clf_bank_real"], bk["clf_bank_ai"], base_rate)
        by_bank = level != "likely real" and r["bank_p"] > clf_p
        match = f" (by bank: comb at {bk['spacings'][r['bank_i']]:.3f} Hz, like {bk['names'][r['bank_i']]})" \
            if by_bank else ""
        verdict = f"bank={bs:.3f} -> {level}{match}: {txt}"
        is_ai = level != "likely real"
    if is_ai:
        verdict += most_like(bundle["model"], x)
    # train_twoband.py: real songs with nothing above 16 kHz were flagged 8-10% of the time, not 5%
    low = f"  (content only up to {top:g} kHz: more false alarms)" if top < 16 else ""
    win = how
    return (f"{Path(f).name} [{src_sr / 1000:g} kHz, both bands{win}]: precise={s16:.1f} / {s44:.1f}"
            f"{comb_note(sharp, exact, cut)}  P(AI)={p:.2f}{' ' if joint or bank else ' -> '}{verdict}{low}")


def predict_one_band(f, bundle, args):
    """One fakeprint at the settings the model was trained with (or the command-line settings)."""
    sr = bundle.get("sr", args.sr) if bundle else args.sr
    fmin = bundle.get("fmin", args.fmin) if bundle else args.fmin
    fmax = bundle.get("fmax", args.fmax) if bundle else args.fmax
    if args.match_format:  # same 16 kHz, 40 kbps MP3 conversion as every song in our tests
        from audio_io import load_audio_bytes
        from fetch_data import to_match_format
        data, src_sr = read_start(f, args.max_duration)
        audio, sr = load_audio_bytes(to_match_format(data, src_sr), target_sr=sr, max_duration=args.max_duration)
    else:
        audio, sr = load_audio(f, target_sr=sr, max_duration=args.max_duration)
    band_f, fp = fpm.fakeprint(*fpm.average_spectrum_db(audio, sr), fmin, fmax)
    comb, _ = fpm.comb_score(band_f, fp, bump_hz=fpm.BUMP_HZ)
    sharp, exact = fpm.precise_comb(band_f, fp)
    line = f"{Path(f).name}: comb={comb:.2f} precise={sharp:.1f}{comb_note(sharp, exact, SHARPNESS_THRESHOLD)}"
    if bundle is not None:
        cut, model = bundle.get("cutoff", 0.5), bundle["model"]
        p = float(ai_proba(model, fp[None, :])[0])  # one class per family (family_test.py): 1 - P(real)
        line += f"  P(AI)={p:.2f} -> {'AI' if p > cut else 'real'}{most_like(model, fp) if p > cut else ''}"
    return line


def cmd_predict(args):
    load = lambda p: joblib.load(p) if p is not None and Path(p).exists() else None  # noqa: E731
    if args.model is None:
        # 44.1 kHz (or higher) files: the two-band model. Others: the first one-band model that exists
        # (per-family model, one-boundary model, the small one from 'train').
        names = ("model_family.joblib", "model_final.joblib", "model.joblib")
        one_path = next((HERE / n for n in names if (HERE / n).exists()), HERE / names[-1])
        two_path = HERE / "model_twoband.joblib"
        two, one = load(two_path), load(one_path)
    else:
        b = load(args.model)
        two_path, one_path = Path(args.model), Path(args.model)
        two, one = (b, None) if b is not None and "bands" in b else (None, b)
        if b is None:
            print(f"(no model at {args.model})")
    only_two = args.model is not None and two is not None
    if args.joint and (two is None or "joint_cutoff" not in two):
        sys.exit("--joint needs model_twoband.joblib with a joint cut-off (run train_twoband.py)")
    # Classifier or spacing bank: the default for 44.1 kHz files in the window mode, if the model has a bank
    use_bank = (not args.no_bank and not args.joint and not args.whole and two is not None and "bank" in two)
    mode = "first" if args.first_30s else "whole" if args.whole else "windows"
    if two is not None:
        d = two["max_duration"]
        span = (f"first {d:g} s" if mode == "first" else
                f"median over {d:g} s windows, up to {MAX_WINDOWS} (first {d:g} s if the song is under {2 * d:g} s)")
        if mode == "whole" and "whole" in two:
            w = two["whole"]
            span = (f"whole song, up to {w['max_seconds'] / 60:g} min, ranked among "
                    f"{len(w['real_scores']['clf']):,} complete real songs; songs under {w['min_seconds']:g} s: "
                    f"first {d:g} s, ranked among {len(two['real_scores']['clf']):,} real 30 s clips")
    if two is not None and args.joint:
        print(f"(44.1 kHz files: {two_path.name}, both bands, classifier or precise comb with one joint cut-off: "
              f"AI if joint > {two['joint_cutoff']:.3f}, i.e. the song's highest score is above "
              f"{100 * two['joint_cutoff']:.1f}% of real training songs; {span})")
    elif use_bank:
        print(f"(44.1 kHz files: {two_path.name}, both bands; {span}. Classifier or spacing bank "
              f"({len(two['bank']['spacings'])} known comb spacings; --no-bank for the classifier alone). Rank among "
              f"real songs: above {100 * LEVELS[1]:g}% = likely AI, {100 * LEVELS[0]:g}-{100 * LEVELS[1]:g}% = unsure, "
              f"below = likely real.)")
    elif two is not None and "ai_scores" in two:
        print(f"(44.1 kHz files: {two_path.name}, both bands; {span}. Rank among real songs: above "
              f"{100 * LEVELS[1]:g}% = likely AI, {100 * LEVELS[0]:g}-{100 * LEVELS[1]:g}% = unsure, below = likely "
              f"real; the tested AI / real cut-off is {100 * (1 - 0.05):g}%.)")
    elif two is not None:
        print(f"(44.1 kHz files: {two_path.name}, both bands, AI if P(AI) > {two['cutoff']:.2f}; {span})")
    if not only_two:
        if one is None:
            print(f"(no model at {one_path}: showing the comb scores only; run 'train' first for P(AI))")
        else:  # use the settings (and cut-off, if saved) the model was trained with
            args.max_duration = one.get("max_duration", args.max_duration)
            print(f"({'other files' if two is not None else 'model'}: {one_path.name}, "
                  f"AI if P(AI) > {one.get('cutoff', 0.5):.2f}{'; --joint is for 44.1 kHz files only' if args.joint else ''})")
        band = ((one.get("sr", args.sr), one.get("fmin", args.fmin), one.get("fmax", args.fmax)) if one is not None
                else (args.sr, args.fmin, args.fmax))
        if band + (args.max_duration,) != CALIBRATED:
            print("(the precise-comb cut-off is calibrated for --sr 16000 --fmin 1000 --fmax 8000 --max-duration 30 "
                  "--match-format; with other settings, treat it as a rough guide)")
    for f in args.files:
        try:
            if two is not None:
                data, src_sr = read_start(f, read_seconds(two, mode))
                if src_sr >= two["min_sr"]:
                    print(predict_two_band(f, two, data, src_sr, args.joint, mode=mode, base_rate=args.base_rate,
                                           bank=use_bank))
                    continue
                if only_two:
                    print(f"{Path(f).name}: {src_sr / 1000:g} kHz, but {two_path.name} needs "
                          f"{two['min_sr'] / 1000:g} kHz or more (try --model model_family.joblib)")
                    continue
            print(predict_one_band(f, one, args))
        except Exception as exc:  # unreadable file: say so, keep going
            print(f"{Path(f).name}: could not read ({exc})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest", help="check the method on synthetic audio (no data needed)")
    s.add_argument("--n", type=int, default=24, help="clips per class")
    s.add_argument("--duration", type=float, default=12.0, help="seconds per clip")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default=str(HERE / "reports" / "selftest.html"))
    s.set_defaults(func=cmd_selftest)

    t = sub.add_parser("train", help="train and evaluate on folders of real and AI songs")
    t.add_argument("--real", default=str(HERE / "data" / "real"))
    t.add_argument("--ai", default=str(HERE / "data" / "ai"))
    t.add_argument("--model", default=str(HERE / "model.joblib"))
    t.add_argument("--out", default=str(HERE / "reports" / "train.html"))
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="score individual files")
    p.add_argument("files", nargs="+")
    p.add_argument("--model", default=None,
                   help="default: model_twoband.joblib for 44.1 kHz files; for others model_family.joblib, else "
                        "model_final.joblib, else model.joblib (first that exists)")
    p.add_argument("--match-format", action="store_true",
                   help="one-band models: first convert to 16 kHz mono 40 kbps MP3, like every song in our tests "
                        "(the two-band model always converts)")
    p.add_argument("--joint", action="store_true",
                   help="44.1 kHz files: AI if the classifier or the precise comb stands out, with one joint cut-off "
                        "(still 5%% of real songs). Better for new generators with a comb (MiniMax 81%% -> 95%%), "
                        "worse for ones without (Udio 91%% -> 73%%)")
    p.add_argument("--no-bank", action="store_true",
                   help="44.1 kHz files: the classifier alone, without the spacing bank (the default is classifier or "
                        "bank: more AI caught with fewer false alarms on new songs, but songs whose speed was changed "
                        "and generators with no comb, like Lyria, are caught less often; family_bank_test.py)")
    p.add_argument("--first-30s", action="store_true",
                   help="44.1 kHz files: score only the first 30 s (the oldest way)")
    p.add_argument("--whole", action="store_true",
                   help="44.1 kHz files of 60 s or more: one fakeprint of the whole song (up to 8 min) with its own "
                        "cut-off, instead of the median over 30 s windows. Catches more AI (99.7%% vs 95.8%% of new "
                        "songs), but flagged 7.0%% of new real songs vs 3.3%% (wholecheck_test.py)")
    p.add_argument("--base-rate", type=float, default=0.0,
                   help="44.1 kHz files: the share of AI songs where you use the detector (e.g. 0.1); also prints "
                        "a rough chance that each song is AI. Rough: new generators score lower than training ones")
    p.set_defaults(func=cmd_predict)

    # Defaults match the paper's open-source setup (44.1 kHz, 5-16 kHz band).
    # For 16 kHz audio such as the SONICS dataset, use: --sr 16000 --fmin 1000 --fmax 8000
    for parser in (t, p):
        parser.add_argument("--sr", type=int, default=44100, help="resample everything to this rate")
        parser.add_argument("--fmin", type=float, default=fpm.FMIN, help="band start, Hz")
        parser.add_argument("--fmax", type=float, default=fpm.FMAX, help="band end, Hz")
        parser.add_argument("--max-duration", type=float, default=180.0, help="seconds analysed per song")

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

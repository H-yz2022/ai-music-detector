"""Web demo: see an example result, then check your own song for the trace AI music generators leave behind.

  streamlit run app.py

Uses the same models and steps as `detect.py predict` (via demo.analyze; the models are stored as plain numbers in
demo_models\\, so no pickles are needed when the app is hosted). Example songs: examples\\ (make_examples.py).
"""
import json
import tempfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import demo

st.set_page_config(page_title="AI music detector", page_icon=":material/graphic_eq:", layout="centered")

EXAMPLES = Path(__file__).parent / "examples"
LABELS = {"suno": "AI song (Suno)", "human": "Song made by people", "minimax": "AI song (MiniMax, never trained on)"}
CREDITS = (json.loads((EXAMPLES / "credits.json").read_text(encoding="utf-8"))
           if (EXAMPLES / "credits.json").exists() else {})


@st.cache_resource
def models():
    return demo.load_bundle("twoband"), demo.load_bundle("family")


@st.cache_data(max_entries=20, show_spinner=False)
def analyze(data: bytes, name: str):
    two, one = models()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(name).name
        path.write_bytes(data)
        return demo.analyze(path, two, one)


def minutes(s):
    return f"{int(s // 60)}:{int(s % 60):02d}"


def show_verdict(r):
    if r["rank"] is not None:
        detail = f"It scores higher than **{100 * r['rank']:.1f}%** of {r['n_real']:,} human-made songs we tested."
    else:
        detail = (f"AI score {r['p']:.2f} (above {r['cutoff_p']:.2f} means likely AI). This file has a low sample "
                  f"rate ({r['sample_rate'] / 1000:g} kHz), so a simpler check was used.")
    if r["level"] == "likely AI":
        st.error(detail, title="Likely made by AI", icon=":material/smart_toy:")
    elif r["level"] == "unsure":
        st.warning(detail + " That is unusual, but not clear enough to say.", title="Unsure", icon=":material/help:")
    else:
        st.success(detail, title="Likely made by people", icon=":material/person:")


def show_metrics(r):
    with st.container(horizontal=True):
        if r["rank"] is not None:
            st.metric("Scores higher than", f"{100 * r['rank']:.1f}%", border=True,
                      help="The share of human-made songs (never used to train the model) that score lower.")
        st.metric("AI score, P(AI)", f"{r['p']:.2f}", border=True,
                  help="The model's raw score from 0 to 1. It is not a true probability; the comparison with "
                       "human-made songs is easier to read.")
        st.metric("Part of the song checked", minutes(r["seconds"]), border=True,
                  help=f"How: {r['how']}." + (" Up to the first 4 minutes are cut into 30-second parts, each part "
                                                "is scored, and the middle score counts." if r["rank"] is not None
                                                else ""))


def show_fingerprint(r):
    name, freqs, fp, spacing = r["bands"][0]
    df = pd.DataFrame({"Frequency (kHz)": freqs / 1000, "Peak strength": fp})
    x_scale = alt.Scale(domain=[float(df.iloc[0, 0]), float(df.iloc[-1, 0])], zero=False, nice=False)
    line = alt.Chart(df).mark_line(strokeWidth=1).encode(
        x=alt.X("Frequency (kHz):Q", scale=x_scale),
        y=alt.Y("Peak strength:Q", scale=alt.Scale(domain=[0, 1.05])),
        tooltip=[alt.Tooltip("Frequency (kHz):Q", format=".3f"), alt.Tooltip("Peak strength:Q", format=".2f")])
    chart = line
    if spacing > 0:
        teeth = np.arange(np.ceil(freqs[0] / spacing), np.floor(freqs[-1] / spacing) + 1) * spacing / 1000
        if len(teeth) <= 150:
            rules = alt.Chart(pd.DataFrame({"Frequency (kHz)": teeth})).mark_rule(
                color="#c0392b", opacity=0.35, strokeDash=[3, 3]).encode(x=alt.X("Frequency (kHz):Q", scale=x_scale))
            chart = rules + line
    st.altair_chart(chart.properties(height=240),
                    alt=f"Fingerprint of the song from {name}" + (f", peaks repeating every {spacing:.1f} Hz"
                                                                  if spacing else ", no repeating pattern"))
    if r.get("bank_spacing") and r["level"] != "likely real":
        st.caption(f":red[Dashed lines]: the peaks line up with **{r['bank_spacing']:.2f} Hz**, an exact spacing that "
                   "known AI generators use (or a whole-number part of one). This is what decided the answer.")
    elif spacing > 0 and r["level"] == "likely real":
        st.caption(f":red[Dashed lines]: an exact repeating pattern was found, every **{spacing:.1f} Hz** "
                   f"({name} band). AI generators often leave one, but the overall check still points to people.")
    elif spacing > 0:
        st.caption(f":red[Dashed lines]: an exact repeating pattern was found, every **{spacing:.1f} Hz** "
                   f"({name} band). This is typical of AI generators.")
    else:
        st.caption(f"No exact repeating pattern was found ({name} band). The answer comes from the whole shape of "
                   "this line.")


def show_result(r):
    show_verdict(r)
    show_metrics(r)
    st.markdown("**The song's fingerprint**: the tiny peaks left once the overall shape of the spectrum is removed.")
    show_fingerprint(r)


# ---------------------------------------------------------------- header
st.title("Is this song made by AI?")
st.markdown("AI music generators leave a hidden trace in the sound: tiny, **evenly spaced spikes** in its frequency "
            "spectrum, like the teeth of a comb. This tool looks for that trace.")
with st.container(horizontal=True):
    for icon, head, text in ((":material/graphic_eq:", "1 · Listen", "Average the song's spectrum, 30 seconds at a time."),
                             (":material/fingerprint:", "2 · Fingerprint", "Keep only the tiny peaks: the song's fingerprint."),
                             (":material/compare_arrows:", "3 · Compare", "Rank it among 9,378 human-made songs.")):
        with st.container(border=True):
            st.markdown(f"{icon} **{head}**")
            st.caption(text)

# ---------------------------------------------------------------- example result
if CREDITS:
    st.header("See an example", divider="gray")
    keys = [k for k in LABELS if k in CREDITS]
    choice = st.segmented_control("Example song", [LABELS[k] for k in keys], default=LABELS[keys[0]],
                                  key="example", label_visibility="collapsed")
    pick = next((k for k in keys if LABELS[k] == choice), keys[0])
    c = CREDITS[pick]
    with st.container(border=True):
        data = (EXAMPLES / c["file"]).read_bytes()
        st.audio(data, format="audio/mpeg")
        who = f"“{c['title']}” by {c['artist']}" if pick == "human" else f"A {c['source'].split(' (')[0]} song"
        st.caption(f"{who}, a {c['seconds']} s clip. {c['license']}, from [{c['source']}]({c['link']}).")
        with st.spinner("Checking the example ..."):
            example = analyze(data, c["file"])
        show_result(example)

# ---------------------------------------------------------------- the detector
st.header("Check your own song", divider="gray")
with st.container(border=True):
    song = st.file_uploader("Upload a song", type=["mp3", "wav", "flac", "ogg"],
                            help="MP3, WAV, FLAC or OGG. Best: the original file, at least 1 minute long.")
    if song is None:
        st.caption(":material/lock: Your file is only used for this check and is not stored.")
    else:
        st.audio(song)
        with st.spinner("Listening to the song ..."):
            try:
                r = analyze(song.getvalue(), song.name)
            except Exception as exc:  # unreadable or too short
                st.error(f"Sorry, this file could not be read ({exc}).", title="Could not check this song")
                r = None
        if r is not None:
            show_result(r)

# ---------------------------------------------------------------- details
with st.expander("How does it work, and how reliable is it?", icon=":material/info:"):
    st.markdown(
        """
**The idea.** AI music generators build the sound step by step, making it longer at each step ("upsampling").
These steps leave tiny, regular spikes in the average frequency spectrum, a kind of fingerprint. The tool removes
the spectrum's overall shape, keeps only the spikes, and two checks look at them: a simple statistical model
(logistic regression, one boundary per family of generators) and a check for spikes at the exact spacings that
known generators use. Whichever stands out more decides. The method follows Afchar et al., *ISMIR 2025*.

**What the answer means.** The song is compared with human-made songs the model never trained on:
- above 99% of them: **likely made by AI**
- between 90% and 99%: **unsure**
- below 90%: **likely made by people**

**How it reads the song.** It scores up to the first 4 minutes in 30-second parts and takes the middle score,
so one odd part cannot decide the answer.

**How well it works (in our tests).** On songs it was never trained on, it caught 100% of new Suno songs and 96%
of MiniMax songs, and wrongly flagged about 5% of 1,454 human-made songs. Some generators leave almost no trace:
Google's Lyria is caught only rarely. Generators it has never seen are caught about two thirds of the time. So a
single result is a hint, not proof.

**What can fool it.** Changing the speed of a song by about 1%, very low-quality files, or a brand-new generator.
"""
    )
st.caption("Research project: an AI music detector based on fakeprints "
           "([code on GitHub](https://github.com/H-yz2022/ai-music-detector)). Your file is not stored.")

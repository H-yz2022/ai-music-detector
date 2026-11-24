"""Web demo: upload a song, see whether it is likely made by AI, and the fingerprint the answer comes from.

  streamlit run app.py

Uses the same models and steps as `detect.py predict` (via demo.analyze; the models are stored as plain numbers in
demo_models\\, so no pickles are needed when the app is hosted).
"""
import tempfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import demo

st.set_page_config(page_title="AI music detector", page_icon=":material/graphic_eq:", layout="centered")


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


st.title("Is this song made by AI?")
st.markdown(
    "Upload a song and this tool checks it for a hidden trace that AI music generators leave behind: tiny, "
    "evenly spaced spikes in the sound's frequency spectrum."
)

song = st.file_uploader("Upload a song", type=["mp3", "wav", "flac", "ogg"],
                        help="MP3, WAV, FLAC or OGG. Best: the original file, at least 1 minute long.")

if song is None:
    st.info("Your file is only used for this check and is not stored.", icon=":material/lock:")
    st.stop()

st.audio(song)
with st.spinner("Listening to the song ..."):
    try:
        r = analyze(song.getvalue(), song.name)
    except Exception as exc:  # unreadable or too short
        st.error(f"Sorry, this file could not be read ({exc}).", title="Could not check this song")
        st.stop()

# The answer
if r["rank"] is not None:
    detail = f"It scores higher than **{100 * r['rank']:.1f}%** of {r['n_real']:,} human-made songs we tested."
else:
    detail = (f"AI score {r['p']:.2f} (above {r['cutoff_p']:.2f} means likely AI). This file has a low sample rate "
              f"({r['sample_rate'] / 1000:g} kHz), so a simpler check was used.")
if r["level"] == "likely AI":
    st.error(detail, title="Likely made by AI", icon=":material/smart_toy:")
elif r["level"] == "unsure":
    st.warning(detail + " That is unusual, but not clear enough to say.", title="Unsure",
               icon=":material/help:")
else:
    st.success(detail, title="Likely made by people", icon=":material/person:")

with st.container(horizontal=True):
    if r["rank"] is not None:
        st.metric("Scores higher than", f"{100 * r['rank']:.1f}%", border=True,
                  help="The share of human-made songs (never used to train the model) that score lower than this song.")
    st.metric("AI score, P(AI)", f"{r['p']:.2f}", border=True,
              help="The model's raw score from 0 to 1. It is not a true probability; the comparison with "
                   "human-made songs is easier to read.")
    st.metric("Part of the song checked", minutes(r["seconds"]), border=True,
              help=f"How: {r['how']}. Up to the first 4 minutes are cut into 30-second parts, each part is scored, "
                   "and the middle score counts.")

# The chart: the song's fingerprint in the main band
name, freqs, fp, spacing = r["bands"][0]
st.subheader("The song's fingerprint")
st.markdown(
    "This line shows the small peaks left in the sound once the overall shape of the spectrum is removed. "
    "In music made by AI, these peaks often **repeat at an exact, even spacing**, like the teeth of a comb. "
    "Music recorded by people usually shows no such pattern."
)
df = pd.DataFrame({"Frequency (kHz)": freqs / 1000, "Peak strength": fp})
line = alt.Chart(df).mark_line(strokeWidth=1).encode(
    x=alt.X("Frequency (kHz):Q", scale=alt.Scale(domain=[float(df.iloc[0, 0]), float(df.iloc[-1, 0])])),
    y=alt.Y("Peak strength:Q", scale=alt.Scale(domain=[0, 1.05])),
    tooltip=[alt.Tooltip("Frequency (kHz):Q", format=".3f"), alt.Tooltip("Peak strength:Q", format=".2f")])
chart = line
if spacing > 0:
    teeth = np.arange(np.ceil(freqs[0] / spacing), np.floor(freqs[-1] / spacing) + 1) * spacing / 1000
    if len(teeth) <= 150:
        rules = alt.Chart(pd.DataFrame({"Frequency (kHz)": teeth})).mark_rule(
            color="#c0392b", opacity=0.35, strokeDash=[3, 3]).encode(x="Frequency (kHz):Q")
        chart = rules + line
st.altair_chart(chart.properties(height=300),
                alt=f"Fingerprint of the song from {name}" + (f", peaks repeating every {spacing:.1f} Hz" if spacing
                                                              else ", no repeating pattern"))
if spacing > 0 and r["level"] == "likely real":
    st.caption(f":red[Dashed lines]: an exact repeating pattern was found, every **{spacing:.1f} Hz** "
               f"({name} band). AI generators often leave one, but the overall check still points to people.")
elif spacing > 0:
    st.caption(f":red[Dashed lines]: an exact repeating pattern was found, every **{spacing:.1f} Hz** "
               f"({name} band). This is typical of AI generators.")
else:
    st.caption(f"No exact repeating pattern was found ({name} band). The answer above comes from the whole shape "
               "of this line.")

with st.expander("How does it work, and how reliable is it?", icon=":material/info:"):
    st.markdown(
        """
**The idea.** AI music generators build the sound step by step, making it longer at each step ("upsampling").
These steps leave tiny, regular spikes in the average frequency spectrum, a kind of fingerprint. The tool removes
the spectrum's overall shape, keeps only the spikes, and a simple statistical model (logistic regression) scores
them. The method follows Afchar et al., *ISMIR 2025*.

**What the answer means.** The song is compared with human-made songs the model never trained on:
- above 99% of them: **likely made by AI**
- between 90% and 99%: **unsure**
- below 90%: **likely made by people**

**How it reads the song.** It scores up to the first 4 minutes in 30-second parts and takes the middle score,
so one odd part cannot decide the answer.

**How well it works (in our tests).** On 1,200 songs it had never seen, it caught about 96% of new songs from
Suno and MiniMax and wrongly flagged about 3% of human-made songs. Generators it never saw are harder: it catches
only about two thirds of them. So a single result is a hint, not proof.

**What can fool it.** Changing the speed of a song by about 1%, very low-quality files, or a brand-new generator.
"""
    )
st.caption("Research project: an AI music detector based on fakeprints. Your file is not stored.")

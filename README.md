# AI Music Detector

Is a song made by AI? This tool answers from the audio alone, by looking for a hidden trace that AI music
generators leave in the sound's frequency spectrum.

**Try it:** [web demo](#web-demo) (upload a song, see the answer and the song's "fingerprint").

It re-implements the method of **"A Fourier Explanation of AI-music Artifacts"** (Afchar, Meseguer-Brocal, Akesbi,
Hennequin, ISMIR 2025, [arXiv:2506.19108](https://arxiv.org/abs/2506.19108)) and tests it much further: on about 20
generators, new real songs from many sources, whole songs, and small changes to the audio.

## How it works

AI generators build the sound from short internal frames and then "upsample" them into audio. Each upsampling step
copies part of the spectrum, which leaves **thin peaks at an exact, even spacing**, like the teeth of a comb.

1. Average the song's spectrum (FFT size 16384, about 2.7 Hz per bin).
2. Remove the overall shape and keep only the small peaks: the **fakeprint**. Two bands are used: 1–8 kHz (of
   16 kHz audio) and 5–16 kHz (of 44.1 kHz audio).
3. Two checks score the fakeprint: a **logistic regression with one boundary per generator family** (17
   generators), and a **spacing bank** that looks for peaks at the exact spacings (decoder frame rates) of 11 known
   generators. Whichever stands out more decides.
4. The score is **ranked among 9,378 real songs** the model never trained on: above 99% = likely AI, 90–99% =
   unsure, below 90% = likely real. A rank is easier to read than a raw probability, which depends on how common
   AI songs are. Songs are scored in 30-second parts (up to 4 minutes) and the middle score counts.

## Results

All numbers below were measured on songs the model never trained on.

| Test | Result |
|---|---|
| 17 generators it was trained on (cross-validated) | **90%** of AI songs caught (classifier), with 5% of real songs flagged |
| New songs: Suno V5.5 / MiniMax / Lyria 3 (MiniMax and Lyria 3 never trained on) | **100% / 96% / 4%** caught |
| 1,454 real songs never used (fresh44, Jamendo, mixtapes, community uploads) | **5.1%** flagged |
| A generator family left out of training, one at a time | **67%** caught on average: Stable Audio 94%, ElevenLabs 90%, Sonauto 55%, but Mureka 21%, Udio 6%, Lyria 4% |
| A 1% speed change | Breaks it (about 5% caught); `--joint` (classifier or precise comb) still catches about 60% |

**What I learned:**
- The trace comes from the generator's architecture, not its training data: even untrained decoders with random
  weights leave it (`python detect.py selftest`).
- The classifier learns the *exact frequencies* of the peaks, so new generators and speed changes can fool it.
  Looking for *even spacing* instead survives these changes.
- Extra features (spectrum shape, stereo, changes over time) looked better on the main tests but raised false
  alarms on new real songs: the model was learning differences between datasets, not AI traces.
- Checking the comb only at the exact spacings of known generators (a "spacing bank") catches more AI with
  fewer false alarms than searching every spacing. With the per-family classifier it became the default
  (new AI songs 79% → 83%, false alarms 5.7% → 5.1%). A speed change moves the spacing, though, and some
  generators (Lyria, Mubert) leave no comb at all.
- Scoring the whole song catches more AI (99.7%), but its cut-off didn't hold on new real songs (7% flagged instead
  of 5%). So the default is the median over 30-second parts.

Every test, with its numbers and what went wrong, is in **[RESEARCH_LOG.md](RESEARCH_LOG.md)**.

## Web demo

```
pip install -r requirements.txt
streamlit run app.py
```

Upload an MP3, WAV, FLAC or OGG file. The page says *Likely made by AI*, *Unsure* or *Likely made by people*, how the
song ranks among real songs, and shows its fingerprint, with dashed lines where peaks repeat at an exact spacing.
Uploaded files are not stored. Three example songs can be tried without uploading anything (`examples/`, made by
`make_examples.py`): "Is It Right?" by Lower Loveday (Jamendo, CC BY-SA), a Suno V5.5 song and a MiniMax Music 3.0
song (both CC BY 4.0, from Hugging Face datasets). Each is the typical case of its group: 11 of 13 candidate human
songs, 8 of 8 Suno songs and 7 of 7 MiniMax songs got the right answer. There is also a Google Colab version
(`demo.ipynb`, see the research log).

## Command line

```
python detect.py selftest                 # no data needed: checks the method on synthetic audio
python detect.py predict song.mp3         # rank among real songs (classifier or spacing bank), comb spacing
python detect.py predict song.mp3 --joint # classifier or comb (better for new generators with a comb)
python detect.py predict song.mp3 --no-bank  # the classifier alone
python detect.py predict song.mp3 --model model_twoband_v1.joblib  # the model before Nov 25, 2026
python detect.py predict song.mp3 --whole # the whole song instead of 30 s parts (more AI caught, more false alarms)
```

## Main files

| File | What it does |
|---|---|
| `fakeprint.py` | The fakeprint, the comb score and the precise comb |
| `detect.py` | Command line: `selftest`, `train`, `predict` |
| `app.py` | Web demo (Streamlit) |
| `examples/`, `make_examples.py` | Three short example songs for the web demo, with credits (`examples/credits.json`) |
| `demo.py`, `demo.ipynb` | Colab demo; exports the models as plain numbers to `demo_models/` |
| `train_twoband.py` | Trains the two-band model used by `predict` |
| `*_test.py` | One script per experiment (see the research log); results in `reports/*.json` |

The audio is not in this repository. The scripts download what they need, keep only the fakeprints, and delete the
audio (see "Data" in the research log).

## Limits

- A result is a hint, not proof: about 3–5% of real songs score like AI.
- Brand-new generators, speed changes and very low-quality files can fool it.
- Built and tested for research, not for judging real artists.

## Credit and license

The method and parameters follow the paper and the authors' code
([deezer/ismir25-ai-music-detector](https://github.com/deezer/ismir25-ai-music-detector), CC BY-NC 4.0). This is an
independent re-implementation for learning and research. The precise comb, synthetic decoder, self-test and the
experiments are my additions. The models were trained on datasets under non-commercial licenses (e.g. SONICS,
CC BY-NC), so they are for **research use only**. All datasets are listed in the research log.

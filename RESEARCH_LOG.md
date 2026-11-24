# AI Music Detector: research log

The full lab notebook: every test, in the order it was run. For a short overview, see [README.md](README.md).

*Clean-up (Nov 24, 2026):* replaced models (`model.joblib`, `model_final`, `model_twoband_v1_old`, `model_twoband_v2`,
`model_twoband_family`), the first audio samples (`data\`, `data_fresh\`), the HTML reports and the demo zip were
deleted. The commands below that made them still work; `predict` uses `model_twoband.joblib` and `model_family.joblib`.

Detects AI-generated music (Suno, Udio and similar) from the audio alone. It
re-implements the method from **"A Fourier Explanation of AI-music Artifacts"**
(Afchar, Meseguer-Brocal, Akesbi, Hennequin · ISMIR 2025 ·
[arXiv:2506.19108](https://arxiv.org/abs/2506.19108)) and adds a training-free
"comb score", plus a self-test that runs with no data.

## The idea in 30 seconds

Music generators turn their internal "latent" frames into audio with a decoder
made of **transposed convolutions**. Each one inserts zeros between samples and
then filters. Inserting zeros copies the spectrum, so the small peak every signal
has at 0 Hz gets copied to every multiple of the decoder's frame rate. The result
is a **comb of thin, evenly spaced peaks** in the spectrum. It comes from the
architecture, not the training data, so every song a model makes carries it.

To find it:
1. Average the song's spectrum over its first 3 minutes (FFT size 16384, about 2.7 Hz per bin).
2. Keep 5–16 kHz, subtract the spectrum's lower envelope, and normalise. The result is the **fakeprint**.
3. **Logistic regression** on the fakeprint (the paper's classifier), plus
4. **Comb score** (my addition): does the fakeprint repeat at a regular spacing?
   This needs no training and also reports the spacing, which is the decoder's frame rate.
   An optional first step removes broad bumps, so only thin peaks count (see the held-out test below).
   A "precise comb" version checks that the peaks repeat at one exact spacing (see "Unseen generators").

## Self-test results (synthetic audio, `python detect.py selftest`)

| Test | Paper's classifier (trained on decoder A) | Comb score (no training) |
|---|---|---|
| Real clips called real | 100% | 100% (0% false alarms) |
| AI clips, decoder A (8,5,4,2), like Encodec | 100% | 100%, spacing 137.8 Hz (expected 137.8) |
| Same AI, **pitch +1 semitone** | **0%** | **100%**, spacing 145.8 Hz (expected 145.9) |
| AI from an **unseen decoder** B (4,4,4,4,2) | **0%** | **100%**, spacing 86.1 Hz (expected 86.1) |

**What this shows:**
- The artifact really does come from the decoder layout alone. The decoders here are untrained, with random weights.
- The trained classifier learns *exact frequencies*, so it breaks when the peaks move. That happens with a pitch shift or a different model version. The paper saw the same thing: Udio's older model was caught only 39.8% of the time, and the authors left pitch shift and resampling to future work.
- Looking for *regular spacing* instead of fixed frequencies survives both changes, and it tells you the model's frame rate.

**Caveat:** the synthetic AI clips don't sound like music. This tests the
mechanism, not real-world accuracy. The next step is real songs (below).

## Real-song test: real vs Suno vs Udio

**Data:** 100 real Creative Commons clips (FMA) plus 50 Suno and 50 Udio songs (SONICS part 10).
Everything is in the same format (16 kHz mono, about 40 kbps MP3), and only the **first 30 seconds** of each
song is used, because the FMA clips are 30 s long. The band is 1–8 kHz, the paper's SONICS setting.

```
python -I scan_sonics.py --real data\real\fma16k --max-duration 30
python -I detect.py train --sr 16000 --fmin 1000 --fmax 8000 --max-duration 30
```

| | Paper's classifier (logistic regression, 5-fold CV) | Comb score (no training, threshold 0.2) |
|---|---|---|
| Real called real | **85%** | 82% (18 false alarms out of 100) |
| Suno caught | **100%** | 56% |
| Udio caught | **64%** | 0% |
| AUC (1.0 = perfect) | **0.91** (real vs all AI) | 0.80 (real vs Suno), 0.36 (real vs Udio) |

**What this means:**
- **The paper's method works on Suno**, even with only 200 songs and 30-second clips: 100% caught.
  The paper reports about 99% with full songs and about 50k training songs.
- **Udio is the open problem.** Udio songs have no regular comb, and score *below* real songs on the comb test.
  The trained classifier catches only 64% of them. Udio likely uses a different kind of decoder.
  The paper also had trouble with one Udio version (39.8%).
- **The comb score is weaker on short clips than on full songs.** In a first scan of 40 full-length songs
  it found the comb in 20 of 20 Suno songs (median 0.42); on 30 s clips it's 56% (median 0.22). Averaging over less audio leaves more noise.
- **The comb false alarms are not about the spacing.** 14 of the 18 real songs that triggered it have a
  "spacing" of 20–25 Hz, at the bottom of the search range. But ignoring spacings under 30 Hz changes nothing,
  on this sample or the fresh one. These real songs have broad bumps in their fakeprint, and broad bumps line up
  with themselves at many shifts. Suno's spacings cluster at multiples of 25 Hz (25, 50, 75, 125, 200 Hz).
  The held-out test below tries a fix.
- 100 songs per group is small. Treat these as first numbers, not final ones.

Reports: `reports\sonics_scan.html` (comb scan, 3 groups) and `reports\train.html` (classifier).

## Held-out test on fresh songs

```
python -I evaluate.py
```

All settings were chosen on the first sample (`data\`). Then each one was tested **only once** on a fresh sample
(`data_fresh\`): 100 new FMA clips and 100 songs from SONICS part 8 (50 Suno, 50 Udio), none shared with
the first sample. Same setup as above: 16 kHz, 1–8 kHz band, first 30 s.

**New: comb score with bump removal.** Before the self-correlation, subtract a 15 Hz running average from
the fakeprint and keep only what sticks out above it. Broad bumps go away, and thin, evenly spaced peaks stay.
I tried 25 versions on `data\` and kept the one with the best AUC for real vs Suno, a rule fixed before
looking at the results. Each comb's threshold is set on `data\` so that 5% of its real songs are flagged.
(This is stricter than the 0.2 threshold in the table above.)

| Fresh songs | Real called real | Suno caught | Udio caught | AUC |
|---|---|---|---|---|
| Paper's classifier (trained on `data\` only) | **88%** | **100%** | **68%** | **0.94** (real vs all AI) |
| Comb score, old | 89% | 8% | 0% | 0.76 (real vs Suno) |
| Comb score, bumps removed | 91% | 24% | 0% | 0.82 (real vs Suno) |

**What this means:**
- **The paper's classifier holds up on new songs.** It did as well as in the first test, or a little better,
  so the first result was not luck. Udio is still its weak spot.
- **Bump removal helps the comb score on new songs** (AUC 0.76 → 0.82), so broad bumps were part of the
  false-alarm problem. The threshold doesn't carry over perfectly: it flags 9% of fresh real songs, not the planned 5%.
- **On 30 s clips the comb score is still far behind the classifier**, and it finds nothing in Udio.
  So far its strength is robustness (the synthetic test: pitch shift and unseen decoder), not accuracy.

Report: `reports\fresh_test.html`.

## More training data (fakeprints only)

```
python -I build_fakeprints.py sonics --part 10 --exclude data_fresh\ai\sonics
python -I build_fakeprints.py fma --exclude data_fresh\real\fma16k
python -I scale_test.py
python -I encoder_check.py
```

`build_fakeprints.py` downloads a dataset, computes a fakeprint for every song, keeps only the fakeprints
(about 10 KB per song), and deletes the zip. The training pool is **7,894 real FMA clips** and **4,074 AI songs**
from SONICS part 10 (2,223 Suno, 1,851 Udio): 110 MB on disk instead of about 10 GB. The fresh test songs, and
their sister songs from the same prompt, are left out. Same setup as above: 16 kHz, 1–8 kHz band, first 30 s.

`scale_test.py` trains the paper's classifier on more and more of this pool and tests every model on the same
200 fresh songs (`data_fresh\`). Sizes up to 4,000 are the mean of 3 random draws.

| Training songs | Real called real | Suno caught | Udio caught | AUC, all AI | AUC, real vs Udio |
|---|---|---|---|---|---|
| 200 | 84% | 100% | 89% | 0.963 | 0.929 |
| 500 | 88% | 99% | 94% | 0.977 | 0.957 |
| 1,000 | 93% | 99% | 82% | 0.978 | 0.958 |
| 2,000 | 93% | 99% | 85% | 0.988 | 0.979 |
| 4,000 | 95% | 98% | 89% | 0.989 | 0.982 |
| **All 11,968** | **99%** | **98%** | **82%** | **0.993** | **0.987** |

**What this means:**
- **Udio was mainly a data problem.** The AUC for real vs Udio rises from 0.93 to 0.99 as the data grows.
  The paper used about 50k songs; this is 12k.
- **"Udio caught" doesn't rise, because of the 0.5 cut-off.** With all the data the model gets stricter:
  99% of real songs are called real, but 82% of Udio is caught. A lower cut-off would catch more Udio.
  It should be chosen on the training data, not on the test songs (see "Choosing the cut-off" below).
- **Control: it's not detecting the MP3 encoder.** Real and AI songs were MP3-encoded by different programs.
  `encoder_check.py` re-encodes every fresh song with the exact encoder the real training songs went through.
  The result barely changes (AUC 0.991, Suno 96%, Udio 80%). The model's weights are spread evenly over
  1–7.5 kHz, with none at the encoder's top cut-off (7.5–8 kHz).
- **Caveat:** there are only 50 Udio test songs, so one song is 2%. (Artists shared between training and test
  turned out not to matter; see "Artist check" below.)

### Choosing the cut-off

```
python -I cutoff_test.py
```

The 0.5 cut-off isn't tuned for anything. `cutoff_test.py` picks it on the training pool instead.
5-fold cross-validation gives every training song a P(AI) from a model that never saw it (sister songs stay
in the same fold). Then the cut-off is set so that a fixed share of real training songs is flagged.
The two rules, 5% and 1%, were fixed before looking at the fresh songs.

| Cut-off | Real called real (pool / fresh) | Suno caught (pool / fresh) | Udio caught (pool / fresh) |
|---|---|---|---|
| 0.5 (default) | 97% / 99% | 99.8% / 98% | 87% / 82% |
| **5% rule (0.25)** | 95% / 98% | 100% / 100% | **93% / 90%** |
| 1% rule (0.89) | 99% / 99% | 99% / 98% | 71% / 76% |

"Pool" means the 11,968 training songs, cross-validated (AUC 0.993). "Fresh" means the 200 held-out songs
(100 real, 50 Suno, 50 Udio), scored by a model trained on the whole pool.

**What this means:**
- **With the 5% rule, Udio caught on fresh songs goes from 82% to 90%,** and only 2 of the 100 fresh real songs
  are flagged.
- **It's a trade-off.** Allowing only 1% false alarms drops Udio to about 71–76%. Suno is caught at every cut-off.
- The pool numbers are more reliable (1,851 Udio songs instead of 50), and they roughly agree with the fresh ones.

### Artist check

```
python -I fetch_data.py fma-artists        # 358 MB download, keeps a 104 KB track -> artist table
python -I cutoff_test.py --artist-disjoint
```

78 of the 100 fresh real clips are by an artist who also has clips in the training pool (925 clips).
The model could have learned those artists' sound instead of "real vs AI". So this run drops all 925 clips
from training and groups the cross-validation folds by artist.

| Fresh songs, 5% rule | Real called real | Suno caught | Udio caught | AUC |
|---|---|---|---|---|
| Artists shared (before) | 98% | 100% | 90% | 0.993 |
| No shared artists | 98% | 98% | 92% | 0.993 |

**Nothing changes** beyond one song either way, and the cross-validated pool numbers stay the same too
(5% rule: Udio 92% instead of 93%). The model isn't relying on artists it has heard before.

## Unseen generators: 12 other models (Echoes)

```
python -I build_fakeprints.py echoes      # 8.6 GB download, keeps 41 MB of fakeprints
python -I unseen_test.py
```

Everything above was trained only on FMA (real) and SONICS (Suno v2–v3.5, Udio). The
[Echoes dataset](https://huggingface.co/datasets/Octavian97/Echoes) ([arXiv:2603.23667](https://arxiv.org/abs/2603.23667),
CC BY-SA 4.0) has songs from 12 other generators, including diffusion models and Suno v5: 4,463 usable songs.
They get the same format and first 30 s as before, and are never used for training.

Three detectors are compared. Each one's cut-off flags 5% of the real training songs:
- **Classifier:** the paper's logistic regression, with the cut-off from cross-validation (as above).
- **Comb:** the training-free comb score with bump removal.
- **Precise comb** (`precise_comb()` in `fakeprint.py`, also training-free): a decoder repeats at one *exact*
  rate. A spacing just 0.01 Hz off puts the 300th peak 3 Hz away. So for each of the 5 strongest repeat distances
  in the autocorrelation, it tries 2,000 spacings close by and averages the fakeprint at all multiples of each
  (`comb_sharpness()`). Only the exact rate lines up hundreds of peaks at once, which gives one sharp spike. Broad
  shapes in real songs give no spike. The idea came from looking at the comb spacings below, but every setting
  was fixed before each test ran.

| Generator | Classifier | Comb | Precise comb | Either* | Exact spacing (expected) |
|---|---|---|---|---|---|
| Real songs, fresh (false alarms) | 2% | 8% | 3% | 5% | – |
| SONICS Suno (training pool, cross-validated) | 100% | 22% | 97% | 100% | 200.00 Hz |
| SONICS Udio (training pool, cross-validated) | 93% | 0% | 0% | 93% | – |
| ACE-Step | 5% | 78% | **100%** | 100% | 86.13 Hz (44,100 ÷ 512) |
| MusicGen | 41% | 100% | **100%** | 100% | 50.00 Hz (its codec's frame rate) |
| AudioLDM | 88% | 17% | **99%** | 100% | 100.00 Hz (its vocoder: 16,000 ÷ 160) |
| SongGen | 69% | 50% | **96%** | 97% | 50.00 Hz |
| DiffRhythm | 8% | 24% | **94%** | 95% | 21.53 Hz (its VAE: 44,100 ÷ 2,048) |
| ElevenLabs | 61% | 23% | 54% | 67% | 200.00 Hz |
| Brev | 60% | 3% | 49% | 69% | 200.00 Hz |
| Stable Audio 2.0 | 9% | 8% | 48% | 56% | 172.27 Hz (44,100 ÷ 256) |
| Suno v5 | 62% | 1% | 47% | 70% | 200.00 Hz |
| Riffusion (Producer.ai) | 15% | 17% | 43% | 52% | 21.53 Hz |
| Mubert | 5% | 4% | 3% | 9% | – |
| Udio (newer) | 47% | 0% | 0% | 47% | – |
| **All Echoes (4,463)** | **44%** | **33%** | **73%** | **81%** | |

\* Either = the classifier or the precise comb flags the song.

A first version searched only near the comb score's single rough spacing. It missed combs when that guess was off
(e.g. 183 Hz for a 200 Hz comb), and got 67% of Echoes, 81% of SONICS Suno, and 5% false alarms. Searching the
5 strongest repeat distances was the only change. The number 5 was fixed before the run, and the cut-off was set
again on the real training songs (9.21 → 9.23).

**What this means:**
- **The paper's classifier doesn't carry over to new generators.** It catches 44% overall, and almost none of
  ACE-Step, DiffRhythm, Stable Audio or Mubert. Even newer Suno (62%) and Udio (47%) mostly slip past a model
  trained on older versions (98–100% and about 90% before). This matches the self-test: a trained classifier
  learns exact peak positions, and a new decoder puts its peaks somewhere else.
- **The precise comb finds decoders it was never told about, with no training.** It catches 73% of unseen songs,
  more than twice the plain comb (33%), and 94–100% of ACE-Step, MusicGen, AudioLDM, SongGen and DiffRhythm. It also
  lifts SONICS Suno from 22% to 97% and lowers false alarms on fresh real songs from 8% to 3%. Neither of those
  depends on Echoes.
- **The exact spacing is a decoder fingerprint.** Where the frame rate is known, it matches to 0.01 Hz:
  DiffRhythm 44,100 ÷ 2,048, AudioLDM 100, MusicGen 50. The others land on similar exact values: ACE-Step
  44,100 ÷ 512, Stable Audio 2.0 44,100 ÷ 256 (probably one stage of its decoder), and Riffusion the same as
  DiffRhythm. Old Suno, Suno v5, Brev and ElevenLabs all sit at exactly 200.00 Hz, so they may share a similar
  decoder design. This is direct evidence for the paper's theory.
- **Together** the classifier and the precise comb catch 81% of unseen songs, with 5% false alarms on fresh real
  songs. With the plain comb it was 63%, with 10% false alarms.
- **Still missed:** Udio (both versions) and Mubert show no comb at all. Only training on their songs helps there.

## Training on many generators (leave one out)

```
python -I logo_test.py
```

Can the classifier learn to catch a generator it has never seen? For each of the 12 Echoes generators, the
classifier is trained on real FMA clips + SONICS + the **other 11** Echoes generators, then tested on the one left
out. The real clips are split into 12 artist groups. The group held out with each generator sets the cut-off
(5% of those real songs flagged), so the cut-off never sees training songs. The 100 fresh real songs give a
separate false-alarm check. Paper's fakeprint features:

| Held-out generator | Trained on SONICS only | Trained on SONICS + 11 other generators | + precise comb |
|---|---|---|---|
| ACE-Step | 9% | 98% | 100% |
| AudioLDM | 88% | 98% | 100% |
| MusicGen | 41% | 99% | 100% |
| DiffRhythm | 8% | 91% | 98% |
| SongGen | 71% | 90% | 99% |
| Brev | 61% | 80% | 82% |
| Suno v5 | 63% | 76% | 78% |
| ElevenLabs | 60% | 71% | 74% |
| Riffusion (Producer.ai) | 17% | 70% | 75% |
| Stable Audio 2.0 | 11% | 53% | 66% |
| Mubert | 6% | 5% | 9% |
| Udio (newer)* | 47% | **4%** | 4% |
| **All 4,463 songs** | **45%** | **79%** | **83%** |
| Fresh real songs flagged | 3% | 3.5% | 6.5% |

\* For Suno v5 and Udio, leaving the generator out also removed the SONICS Suno / Udio songs (same label).
So those two models had seen no Suno, or no Udio, at all.

**What this means:**
- **Variety in training is the biggest win so far.** A generator the model has never seen is caught 79% of the
  time, not 45%, with the same false-alarm rate. The biggest jumps are for generators with a comb (ACE-Step,
  DiffRhythm, MusicGen, Riffusion, Stable Audio).
- **Shift-proof features didn't help.** Autocorrelation features ("how far apart are the peaks", wherever they sit)
  got 76% alone and 79% together with the fakeprint, the same as the fakeprint alone. Once training covers many
  generators, the paper's features are enough.
- **Udio is only caught when some Udio is in training.** With all Udio left out (SONICS Udio too), only 4% of the
  newer Udio is caught. The SONICS-only model had seen older Udio and caught 47%. Suno is different: with no Suno in
  training at all, Suno v5 is still caught 76%, probably thanks to Brev and ElevenLabs, which share its exact
  200 Hz comb.
- **With the precise comb:** 83% of new-generator songs (81% with the SONICS-only classifier), with 6.5% false
  alarms instead of 5%. Most of the classifier's gain overlaps with what the precise comb already finds. The real
  extra gains are Brev, Suno v5, ElevenLabs, Stable Audio and Riffusion.
- **Caveat:** Echoes and SONICS songs are the first 30 s of full songs (with intros). The FMA clips are 30 s from
  the middle of songs. A trained model could partly learn "song start vs middle". The precise comb can't, since it
  has no training.

## Final model (all 16k songs)

```
python -I train_final.py        # writes model_final.joblib
```

Trained on every song we have: 7,894 real FMA clips and 8,537 AI songs (SONICS + all 12 Echoes generators).
The cut-off comes from 5-fold cross-validation (real clips grouped by artist): P(AI) > 0.84 flags 5% of real songs.

| | Cross-validated (generators in training) | Fresh songs (never trained on) |
|---|---|---|
| Real called real | 95% | 96% |
| Suno (SONICS + v5) | 98% | 100% |
| Udio (SONICS + newer) | 67% | 76% |
| ACE-Step, AudioLDM, DiffRhythm, MusicGen, SongGen | 97–100% | – |
| Riffusion (Producer.ai) | 90% | – |
| Brev, ElevenLabs, Stable Audio 2.0 | 76–82% | – |
| Mubert | 11% | – |
| AUC | 0.970 | 0.982 |

**What this means:**
- `predict` now knows 12 generator families instead of 2.
- **The trade-off is Udio.** The SONICS-only model caught 90% of fresh Udio with the same 5% rule; this one catches
  76%. One linear model has to cover many kinds of artifact now, and Udio, which has no comb, is a smaller share
  of the AI songs (25% instead of 45%).
- Next idea: one boundary per generator family, which gives Udio its own (below).

### One boundary per generator family

```
python -I family_test.py --phase cv      # cross-validation + fresh songs
python -I family_test.py --phase logo    # leave one generator out
python -I family_test.py --phase save    # -> model_family.joblib (the one detect.py predict uses)
```

Same songs, same weights and same folds as the final model. The only change: the classifier is multi-class
(real + 12 families: suno, udio, acestep, ...), so each family gets its own boundary. AI score = 1 − P(real).
`predict` also says which family a song is most like.

| | One boundary (`model_final`) | Per family (`model_family`) |
|---|---|---|
| Udio, cross-validated | 67% | **80%** |
| All AI songs, cross-validated (5% false alarms) | 87% | **90%** |
| Fresh songs: real called real / Suno / Udio | 96% / 100% / 76% | 96% / 100% / **80%** |
| Fresh songs, AUC | 0.982 | **0.989** |
| New generator (leave one out), classifier alone | **79%** | 75% |
| New generator, + precise comb | **83%** | 81% |

**What this means:**
- **For generators in training, per-family boundaries are better**, mostly because Udio gets its own (67% → 80%).
  ACE-Step, AudioLDM and MusicGen go to 100%.
- **For a brand-new generator, one shared boundary is a little better** (79% vs 75%; 83% vs 81% with the precise
  comb). The losses are Stable Audio, ElevenLabs and DiffRhythm. A boundary drawn for each known family fits new
  families a bit less well.
- So the choice depends on the job: known platforms (Suno, Udio) → `model_family`; new, unknown generators →
  `model_final` + the precise comb. For files below 44.1 kHz, `predict` uses `model_family` by default, because
  it also prints the precise comb, which covers most new generators with a comb. Add `--model model_final.joblib`
  for the other. (44.1 kHz files now get the two-band model, below.)

## 16 kHz vs 44.1 kHz (the paper's main setting)

```
python -I build_fakeprints.py fma --exclude data_fresh\real\fma16k --out features_hq\fma_small.npz --sr 44100 --fmin 5000 --fmax 16000 --match-sr 44100 --mp3-level 0.4 --spectrum
python -I build_fakeprints.py echoes --out features_hq\echoes.npz --sr 44100 --fmin 5000 --fmax 16000 --match-sr 44100 --mp3-level 0.4 --spectrum
python -I hq_test.py
```

Everything above used 16 kHz audio and the 1–8 kHz band, because SONICS only exists at 16 kHz. FMA and Echoes are
44.1 kHz originally, so this test rebuilds them at 44.1 kHz with the paper's 5–16 kHz band. Every song is first
re-encoded to the same 192 kbps MP3, which keeps content up to 20 kHz. Same 12,357 songs in both setups (7,894
real, 4,463 Echoes), no SONICS. All cut-offs flag 5% of real songs. The downloads (16 GB) were deleted afterwards;
`features_hq\` keeps each song's 5–16 kHz spectrum (174 MB).

| Generator | Content up to | Cross-validated 16k / 44.1k | Leave one out 16k / 44.1k | Precise comb 16k / 44.1k |
|---|---|---|---|---|
| ACE-Step | 16 kHz | 100% / 100% | 100% / 74% | 100% / 100% |
| AudioLDM | 9 kHz | 100% / 100% | 99% / 99% | 99% / 19% |
| Brev | 16 kHz | 88% / **97%** | 89% / 95% | 49% / **82%** |
| DiffRhythm | 16.5 kHz | 99% / 100% | 96% / 71% | **94%** / 43% |
| ElevenLabs | 16 kHz | 86% / 91% | 76% / 67% | 54% / 65% |
| Mubert | 16 kHz | 12% / 22% | 3% / 9% | 3% / 1% |
| MusicGen | 15.5 kHz | 100% / 100% | 100% / 98% | 100% / 90% |
| Riffusion (Producer.ai) | 16 kHz | 95% / 96% | **86%** / 33% | **43%** / 6% |
| SongGen | 9 kHz | 99% / 100% | 93% / 90% | **96%** / 25% |
| Stable Audio 2.0 | 16 kHz | 86% / 92% | 61% / 38% | 48% / 52% |
| Suno v5 | 16 kHz | 92% / **99%** | 86% / **99%** | 47% / **95%** |
| Udio (newer) | 16 kHz | 19% / **83%** | 4% / 2% | 0% / 0% |
| **All 4,463** | | **88% / 95%** | **83% / 73%** | **73% / 49%** |

(Real songs have content up to 18 kHz. Udio here has only its 300 Echoes songs in training, since there is no
SONICS, which is why its 16 kHz number is lower than in earlier tables.)

**What this means:**
- **For generators in training, 44.1 kHz is clearly better:** 88% → 95%. The biggest jump is **Udio: 19% → 83%**
  with only 300 Udio songs in training. Udio's artifact is in the high band, which the 16 kHz setup can't see.
  Suno v5, Brev, ElevenLabs and Stable Audio improve too.
- **For a brand-new generator, 16 kHz is better:** 83% vs 73% when left out. ACE-Step, DiffRhythm, Riffusion and
  Stable Audio drop a lot at 44.1 kHz.
- **Different decoders put their comb in different bands.** The 200 Hz combs (Suno v5, Brev, ElevenLabs) are much
  clearer at 5–16 kHz (Suno v5: 47% → 95%). The 21.53 Hz combs (DiffRhythm, Riffusion) are clearer at 1–8 kHz.
  AudioLDM and SongGen have no content above 9 kHz, so 44.1 kHz can't see their comb at all.
- **Check: it's not the MP3 encoding.** Most AI songs stop at 16 kHz and real songs at 18 kHz, right at the top
  of the band. But the 44.1 kHz model's weights are spread evenly (5% at 15.5–16 kHz, the same as that band's
  share of the bins). Suno, Udio and Brev fall off up high like real songs (−15 to −17 dB). And with the band
  stopped at 14 kHz, the results barely change (cross-validated 94%, Udio 78%).
- **So the two bands complement each other.** The next step is to use both together.

### Both bands together

```
python -I twoband_test.py
```

Same 12,357 songs. The classifier gets both fakeprints side by side (1–8 kHz from the 16 kHz audio, 5–16 kHz from
the 44.1 kHz audio). To combine scores without adding false alarms, each score becomes "higher than what share of
real songs", a song's combined score is its highest one, and one cut-off flags 5% of real songs. In leave-one-out,
half of the held-out real songs set the cut-off and the other half measure false alarms.

| Generator | Cross-validated, both bands | Leave one out: 16k / 44.1k / **both** | Precise comb, both bands |
|---|---|---|---|
| ACE-Step | 100% | 100% / 74% / **99%** | 100% |
| AudioLDM | 100% | 99% / 99% / **100%** | 97% |
| Brev | 99% | 89% / 95% / **99%** | 81% |
| DiffRhythm | 100% | 96% / 71% / **99%** | 91% |
| ElevenLabs | 94% | 76% / 67% / **81%** | 60% |
| Mubert | 23% | 3% / 9% / **9%** | 3% |
| MusicGen | 100% | 100% / 98% / **100%** | 97% |
| Riffusion (Producer.ai) | 99% | 86% / 33% / **90%** | 39% |
| SongGen | 100% | 93% / 90% / **95%** | 93% |
| Stable Audio 2.0 | 96% | 61% / 38% / **60%** | 53% |
| Suno v5 | 100% | 86% / 99% / **100%** | 90% |
| Udio (newer) | 91% | 4% / 2% / **4%** | 0% |
| **All 4,463** | **96%** (16k 88%, 44.1k 95%) | **83% / 73% / 86%** | **76%** (16k 73%, 44.1k 49%) |
| Real songs flagged | 5% | 5% / 5% / **4.7%** (measured) | 5% |

**What this means:**
- **Both bands together is the best detector so far.** It catches 96% of songs from generators in training, and
  **86% from a generator it has never seen** (83% with 16 kHz alone, 73% with 44.1 kHz alone). False alarms are
  4.7%, measured on real songs that were not used to set the cut-off.
- **It keeps the strengths of each band:** DiffRhythm, Riffusion and ACE-Step (seen best at 1–8 kHz), plus
  Suno v5, Brev and Udio (seen best at 5–16 kHz).
- **The precise comb no longer adds detection.** Sharing one 5% false-alarm budget, classifier + comb gets 94%
  (cross-validated) and 83% (leave one out), a bit below the classifier alone, because the two-band classifier
  already catches what the comb catches. The comb is still useful to *explain* a verdict with its exact spacing.
- **Still missed:** Udio when no Udio is in training (4%), Mubert (9%), and Stable Audio as a new generator (60%).
- These tables have no SONICS, so they don't match the earlier 16 kHz tables one to one.

### Two-band model for `predict`

```
python -I train_twoband.py      # -> model_twoband.joblib (this first v1 is now model_twoband_v1_old.joblib)
```

The same classifier, trained on all 12,357 songs. The cut-off comes from 5-fold cross-validation (real clips
grouped by artist): **P(AI) > 0.126** flags 5% of real songs. Cross-validated catch rate 96.2% (AUC 0.988), the
same as the test above. The precise-comb cut-offs (5% of real songs) are saved too: 9.23 for 1–8 kHz, 10.33 for
5–16 kHz.

`detect.py predict` now uses it for every file at 44.1 kHz or more. It makes both fakeprints from the first 30 s,
after the same MP3 conversions as the training songs (16 kHz at 40 kbps, and 44.1 kHz at 192 kbps). Files below
44.1 kHz still get `model_family`. A check on synthetic audio showed that predict makes the same fakeprints as
`build_fakeprints.py` (largest difference 0.0002, from storage rounding).

**One thing to know:** real songs with little content above 16 kHz (old recordings, low-bitrate files) are flagged
more often: 8–10% instead of 5% (922 of the 7,894 real clips). `predict` adds a note for such files.

### New 44.1 kHz songs, never trained on

```
python -I fresh44_test.py
```

728 new songs. The script reads only about the first 30 s of each song straight from the server (663 MB in
total) and keeps only the fakeprints (`features_hq\unseen\fresh44.npz`). The cut-offs are the saved ones, so
nothing was tuned on these songs.

| Songs | Count | What it tests | Two-band (44.1 kHz+ files) | 16 kHz model | `predict` | Exact comb |
|---|---|---|---|---|---|---|
| Real, FMA large (artists not in training) | 298 | false alarms, same site as training | 6.0% flagged | 6.0% | 6.0% | 14% |
| Real, Jamendo (jam-alt) | 79 | false alarms, a new music source | 5.1% flagged | 1.3% | 5.1% | 6% |
| Suno V5.5 (May 2026) | 150 | a newer version (training has v5) | **100%** | 90% | **100%** | 95%, at 200.00 Hz |
| MiniMax Music 3.0 | 150 (112 at 44.1 kHz) | a new generator | **81%** | 35% | 68% | 84%, at 689.06 Hz |
| Lyria 3 (source not verified) | 51 | a new generator | 6% | 14% | 6% | 2% |

`predict` = what `detect.py predict` does: the two-band model for files at 44.1 kHz or more, the 16 kHz model
for the rest. "Exact comb" = the precise comb stands out in either band.

**What this means:**
- **False alarms hold up on new real songs:** 22 of 377 (5.8%), close to the 5% target, including Jamendo, a
  music source the model has never seen.
- **A newer Suno version is fully caught** (16 kHz model: 90%).
- **MiniMax, a brand-new generator, is caught 81% of the time with both bands vs 35% with the 16 kHz model.**
  Its comb sits at exactly 689.06 Hz = 44,100 / 64 (now in `predict`'s list) and shows in 95% of its
  44.1 kHz files, more than the classifier catches. So for a new generator the precise comb can still add
  detection (checked below).
- 38 MiniMax files are only 16–32 kHz. `predict` sends them to the 16 kHz model, which catches 29% of them.
- **Lyria 3 is missed completely:** no comb, and P(AI) as low as real songs. Like Udio and Mubert, only training
  on it can help. (The Lyria set has no description, so its source is not verified.)
- Real songs with little content above 16 kHz were flagged 3.6% here (55 songs), so the higher rate seen in
  training did not show up in this small sample.

### Classifier + comb with one joint cut-off, on the new songs

`train_twoband.py` now also saves the real training songs' scores, so the joint rule from "Both bands together"
can be used on new songs. Each score (classifier, comb at 1–8 kHz, comb at 5–16 kHz) becomes "higher than what
share of real training songs", the highest one counts, and one cut-off flags 5% of real training songs. The rule
was fixed before this test, so nothing was tuned on the new songs.

| | Classifier alone | Classifier + comb (joint) |
|---|---|---|
| New real songs flagged (377) | 5.8% | 6.1% |
| Suno V5.5 | 100% | 100% |
| MiniMax Music 3.0 (new generator) | 81% | **95%** |
| Lyria 3 | 6% | 4% |
| Training generators, cross-validated | **96%** | 94% (Udio 91% → 73%, Mubert 23% → 13%) |

**What this means:**
- **Same false alarms, and a big gain on a new generator that has a comb:** MiniMax goes from 81% to 95%.
- **Generators without a comb lose:** Udio drops from 91% to 73% (with Udio in training), because the comb uses
  part of the 5% false-alarm budget. In the leave-one-out test above, the joint rule was also a bit lower on
  average (83% vs 86%).
- So it is a trade-off: known platforms such as Udio → the classifier alone (what `predict` uses by default);
  screening for new, unknown generators → the joint rule (`predict --joint`, 44.1 kHz files only).

## More AI songs: Udio and four new generators

```
python -I more44_test.py                  # download (first ~30 s of each song only), then 'before'
python -I more44_test.py --phase after    # 70% into training, 30% held out; leave one new generator out
```

2,762 new AI songs, all 44.1 or 48 kHz. 3.08 GB was streamed, and only fakeprints were kept (83 MB in
`features_hq\more44\`):

| Generator | Songs | From |
|---|---|---|
| Udio | 1,000 (100 from each of 10 shards) | blanchon/udio_dataset (CC0) |
| Mureka v9 | 662 | MUSIC8K (CC BY 4.0) |
| HeartMuLa | 500 | MUSIC8K (CC BY 4.0) |
| Sonauto | 500 (50 from each of 10 places) | ai-music/ai-music-deduplicated (MIT) |
| Lyria Pro 3 | 100 (made through Google's API) | HAIM (CC BY-NC 4.0) |

**Before: the current two-band model, these songs never trained on**

| | Udio | Mureka | HeartMuLa | Sonauto | Lyria Pro 3 |
|---|---|---|---|---|---|
| Two-band classifier | **97%** | 17% | **100%** | 27% | 14% |
| Joint (classifier or comb) | 94% | 8% | 100% | 20% | 7% |
| 16 kHz model | 78% | 8% | 99% | 14% | 15% |
| Exact comb | 0% | 2% | **100%, at 125.00 Hz** | 21% | 3% |

- New songs from Udio, a generator already in training, are caught 97% of the time.
- HeartMuLa is caught fully thanks to its comb: 125.00 Hz at 1–8 kHz, and steps of 25 Hz at 5–16 kHz. That
  spacing is now in `predict`'s list.
- Mureka, Sonauto and Lyria Pro 3 have almost no comb, and the model misses most of them.

**After: training with 70% of each new source (Udio and Sonauto split by shard), tested on the other 30%**

| | Udio | Mureka | HeartMuLa | Sonauto | Lyria Pro 3 |
|---|---|---|---|---|---|
| Before | 98% | 16% | 100% | 35% | 13% |
| **After** | 97% | **37%** | 100% | **84%** | **37%** (30 songs) |
| Left out: trained on the other new sources, never this one | 92% | 26% | 100% | 31% | 23% |

Then the final model on everything (`model_twoband_v2.joblib`, 15,119 songs). Cut-off P(AI) > 0.704 for 5% real
false alarms; cross-validated AUC 0.968 (was 0.988).

| | v1 (`model_twoband.joblib`, default) | v2 (`model_twoband_v2.joblib`) |
|---|---|---|
| Real songs flagged (377 new, fresh44_test) | **5.8%** | 6.6% |
| Suno V5.5 / MiniMax (new songs) | **100%** / **81%** | 97% / 80% |
| Lyria 3 (the other Lyria set, never trained on) | 6% | **24%** |
| Sonauto / Mureka / Lyria Pro 3 | ~30% / 17% / 14% | **~80% / ~38% / ~40%** |
| Udio (Echoes) / ElevenLabs / Stable Audio, cross-validated | **91% / 94% / 96%** | 79% / 87% / 89% |

**What this means:**
- **Training data helps generators with no comb, but not enough for all.** Sonauto goes from about 30% to
  80%. Mureka and Lyria Pro 3 reach only about 40%, even when trained on.
- **Training on Lyria Pro 3 also helps another Lyria set** (6% → 24%), so it learns something real about Lyria.
- **A wider mix alone helps only a little:** generators left out of training are caught only a few points more than before.
- **One straight boundary is now stretched too far.** With 17 generators, the cut-off moved from 0.126 to 0.704,
  the AUC fell from 0.988 to 0.968, and several known generators dropped (Echoes Udio 91% → 79%). False alarms
  went up a little too.
- So `predict` keeps **v1** as the default (best on Suno and Udio, fewer false alarms). Use
  `--model model_twoband_v2.joblib` for wider coverage (Sonauto, Lyria, Mureka).
- The next fix to try is one boundary per generator family (like `model_family` at 16 kHz). That should
  keep the old generators while learning the new ones (next section).

### One boundary per generator family, both bands

```
python -I more44_test.py --phase after --family
python -I train_twoband.py --extra-dir features_hq\more44 --family --out model_twoband_family.joblib
```

The same logistic regression on the same two fakeprints, but multi-class: "real" plus one class per generator
family (16 families; the two Udio sets count as one family). AI score = 1 − P(real). So each family gets its own
boundary instead of sharing one, which is what helped Udio at 16 kHz.

**Held-out 30% of the new songs (trained on 70%):**

| | Udio | Mureka | HeartMuLa | Sonauto | Lyria Pro 3 |
|---|---|---|---|---|---|
| v1 (not trained on them) | 98% | 16% | 100% | 35% | 13% |
| One boundary, after | 97% | 37% | 100% | 84% | 37% |
| **Per family, after** | **98%** | **43%** | 100% | **91%** | **40%** |
| Per family, left out (never seen) | 97% | 24% | 99% | 22% | 18% |

**The three final models (each trained on everything it uses):**

| | v1 (default) | v2: one boundary, 17 generators | **Per family, 17 generators** |
|---|---|---|---|
| Real songs flagged (377 new, fresh44_test) | 5.8% | 6.6% | **4.0%** |
| Suno V5.5 (new songs) | **100%** | 97% | **100%** |
| MiniMax (never trained on) | **81%** | 80% | 72% (joint: 94%) |
| Lyria 3 (other Lyria set, never trained on) | 6% | **24%** | 20% |
| Sonauto / Mureka / Lyria Pro 3 (cross-validated) | ~30% / 17% / 14% (not trained on) | 76% / 39% / 45% | **84% / 49%** / 42% |
| Echoes Udio / ElevenLabs / Stable Audio (cross-validated) | **91% / 94% / 96%** | 79% / 87% / 89% | 88% / 90% / 93% |
| All AI, cross-validated | 96% (12 generators) | 88% (17) | **90%** (17) |

**What this means:**
- **Per family is the best model for generators it has seen.** It keeps the old generators almost as well as v1,
  learns the new ones better than v2 (Sonauto 84%, Mureka 49%), and has the fewest false alarms (4.0%).
- **It is a bit weaker on brand-new generators**, like at 16 kHz: MiniMax 72% vs 81%, and new generators left out of
  training are caught less often than with one boundary. `--joint` makes up for it on MiniMax (94%).
- Mureka and Lyria are still the hard ones: under 50% even when trained on.
- `predict` still uses v1 by default. The per-family model is `--model model_twoband_family.joblib`; it also says
  which family a song is most like.

**Why MiniMax drops with per family:** of the 112 MiniMax songs, the per-family model puts 57 closest to "real", 34
closest to Sonauto and 21 closest to Stable Audio. It asks "is this one of the families I know?", and MiniMax is
none of them, so "real" often wins (the 31 missed songs: P(real) 0.87 on average). One boundary asks "AI-like or
real-like?", which carries over better to a new generator. 106 of the 112 have a precise comb, so `--joint` still
catches them.

### A non-linear model: LightGBM (boosted decision trees)

```
python -I nonlinear_test.py --models lgbm lr family     # ~1 hour; results in reports\nonlinear_test.json
```

Logistic regression adds up one weight per frequency bin: one straight boundary (or one per family). LightGBM
builds 500 small decision trees, so it can use rules like "peak at 7,000 Hz **and** at 9,000 Hz". Same songs (15,119),
same two fakeprints, balanced classes, settings fixed before running. The bins it uses most are exact comb peaks:
5,800, 9,000, 3,200, 7,250, 7,000 Hz and 5,512.5 Hz (= 44,100 / 8, MiniMax's comb).

| | LR one boundary (v2) | LR per family | LightGBM |
|---|---|---|---|
| All AI, cross-validated | 88% | 90% | **96%** |
| Mureka / Lyria Pro 3 / Sonauto, cross-validated | 39% / 45% / 76% | 49% / 42% / 84% | **94% / 80% / 97%** |
| All AI, family never seen (leave one family out) | 64% | 65% | **84%** |
| Udio / Sonauto / Mureka / Lyria, never seen | 11% / 33% / 20% / 42% | 21% / 23% / 17% / 38% | **67% / 86% / 85% / 91%** |
| MusicGen / Producer, never seen | **100% / 85%** | 99% / 81% | 74% / 62% |
| Real FMA artists not in training, flagged | 4.7% | 4.9% | 6.0% |
| **New real songs (fresh44), flagged** | 6.6% | **4.0%** | **38%** |
| Suno V5.5 / MiniMax / Lyria 3 (fresh44) | 97% / 80% / 24% | 100% / 72% / 20% | 100% / 100% / 86% |
| Same, cut-off re-set so new real songs get ~5% (avg of 20 half splits) | — | 100% / 76% / 26% | 96% / 82% / 28% |

**What this means:**
- **LightGBM is much better at AI songs**, even from generator families it never saw (84% vs 65%).
- **But it is not usable yet: it flags 38% of new real songs.** Real FMA artists held out from training are fine
  (6%), so it learned what *our* real songs look like (FMA small: 30 s clips from one archive) too closely. The new
  real songs (other FMA tracks' first 30 s, and Jamendo) don't fit that look, and the trees call them AI. The
  fakeprints of both real sets have the same average values, so this is not a processing bug. Logistic regression is
  a smoother rule and carries over (4–7%).
- With the cut-off re-set so new real songs get the same ~5% false alarms, LightGBM's gain on fresh44 is gone
  (MiniMax 82% vs 76%, Lyria 3 28% vs 26%).
- The fix to try: **more varied real songs in training** (different archives, full tracks, different positions in
  the song), then test again on real songs from a source that was not used in training.
- fresh44 has now also checked LightGBM; its real songs were used to explain the false alarms above.

### More varied real songs, and six models

```
python -I more_real.py      # 1,500 real songs, 1.55 GB streamed, 43 MB of fakeprints kept
python -I nonlinear_test.py --extra-real features_hq\more_real --hold-real etree --tests cv fresh --models lr family lgbm mlp rf svm --out reports\nonlinear_morereal_holdetree.json
python -I nonlinear_test.py --extra-real features_hq\more_real --models lr family lgbm mlp rf svm --out reports\nonlinear_morereal.json
```

New real training songs (44.1 kHz and up only): 595 FMA-large clips (artists in neither fma_small nor fresh44, all
genres: Avant-Garde, Noise, Jazz, Lo-Fi, …), 440 netlabel songs and 449 live-concert recordings from the Internet
Archive (30 s from 40% into the song). Jamendo stays out, so fresh44's Jamendo songs are still a real source never
trained on. Three more models were added: a small neural network (MLP, 256 hidden units), a random forest (500
trees) and an RBF SVM (2,000 Nystroem features + logistic regression). Settings fixed before running.

| | LR one boundary | LR per family | LightGBM | Neural net | Random forest | RBF SVM |
|---|---|---|---|---|---|---|
| AI caught, cross-validated (17 generators) | 88% | 90% | **94%** | 89% | 87% | 89% |
| AI caught, family never seen | 63% | 64% | **74%** | 65% | 58% | 65% |
| **New real songs flagged (fresh44, 377)** | 5.3% | **4.2%** | 16% | 4.5% | 6.9% | 8.0% |
| Live concerts flagged, never trained on (449) | 7.1% | 5.1% | 14% | 6.7% | 4.0% | **2.4%** |
| Suno V5.5 / MiniMax / Lyria 3 (fresh44) | 97 / 80 / 16% | 99 / 70 / 16% | 100 / 100 / 75% | 99 / 84 / 14% | 98 / 96 / 37% | 99 / 97 / 26% |
| Mureka / Lyria Pro 3, cross-validated | 39 / 42% | 46 / 47% | **84 / 66%** | 42 / 35% | 66 / 43% | 65 / 54% |

(All rows but "live concerts" use every real song in training. The live-concert row comes from the run that kept
them out of training. Joint rule on fresh44 MiniMax: 94–98% for every model.)

**What this means:**
- **Varied real songs helped LightGBM, but not enough:** new real songs flagged 38% → 16%, still three times too
  many. Its never-seen-family score also fell (84% → 74%), so part of that gain was the shortcut "doesn't look like
  FMA small → AI".
- **Random forest and RBF SVM** catch brand-new comb generators better (MiniMax 96–97%), but flag more new real songs
  (7–8%) and do worse on some never-seen families (MusicGen, Producer).
- **The neural network is about the same as logistic regression.**
- **Per-family logistic regression is still the best all-round model:** fewest false alarms, and close to the best on
  the generators it knows. The simple model wins because the hard part is the data, not the model. Generators with
  no comb (Mureka, Lyria, Mubert) are hard for every model, and the fancier models mostly learn the training songs
  more closely.
- Adding the varied real songs to the LR models changes little (one boundary: new real songs flagged 6.6% → 5.3%;
  per family: 4.0% → 4.2%).
- fresh44 has now been used for these checks too.

### v1 retrained with the varied real songs (the new default)

```
python -I train_twoband.py --extra-dir features_hq\more_real      # -> model_twoband.joblib (predict's default)
```

The same recipe as v1 (one boundary, the 12 Echoes generators), plus the 1,484 new real songs at 44.1 kHz or more
(9,378 real songs in all). The old v1 is kept as `model_twoband_v1_old.joblib`. To check it on a real source it never
saw, a second copy was trained without the live concerts (etree). Nothing was tuned on the test songs.

| | Old v1 | **New v1** | New v1 without live concerts |
|---|---|---|---|
| Cut-off (5% of real training songs) | P(AI) > 0.126 | P(AI) > 0.100 | P(AI) > 0.108 |
| AI caught, cross-validated (12 generators) | 96.2% | 96.3% | 96.2% |
| Real songs flagged, cross-validated: FMA small / FMA large / netlabels / live | 5% / – / – / – | 4.7 / 5.5 / 8.4 / 6.9% | 4.9 / 4.7 / 7.5 / – |
| **New real songs flagged (fresh44, 377)** | 5.8% | 6.1% | 5.8% |
| Live concerts flagged, never trained on (449) | 7.1% | (trained on) | **7.1%** |
| Suno V5.5 / MiniMax / Lyria 3 (fresh44) | 100 / 81 / 6% | 100 / **86** / 4% | 100 / 87 / 4% |
| Udio / Mureka / HeartMuLa / Sonauto / Lyria Pro 3 (more44, never trained on) | 97 / 17 / 100 / 27 / 14% | 98 / 16 / 100 / 28 / 15% | 97 / 16 / 100 / 28 / 15% |

**What this means:**
- **The new v1 is about as good as the old one, and a bit better on MiniMax (81% → 86%).** AI songs are caught just
  as well, and false alarms on new real songs are the same (23 vs 22 of 377).
- **The 5% cut-off now rests on more kinds of real music** (other FMA genres, netlabels, live recordings), so music
  like that is no longer an unknown. Before, the old v1 flagged 6.4% of these 1,484 songs.
- **But more real songs did not make it better on a real source it has never seen:** live concerts are flagged 7.1%
  with or without the other new real songs. Netlabels stay the hardest real source (7.5–8.4%, even when trained on).
- So `predict` now uses the new v1 by default (`--model model_twoband_v1_old.joblib` for the old one). The tables
  above this section were made with the old v1.

## Robustness: small changes to a song

```
python -I robust_test.py      # 220 songs, 233 MB read, ~19 min; reports\robust_test.json
```

Can someone hide an AI song by changing it a little? 220 songs never trained on: 100 real (70 FMA large, 30
Jamendo, from fresh44), 40 Suno V5.5, 40 MiniMax 3.0 (from fresh44) and 40 Udio (a shard more44 did not use). Only
the first 30 s of each was read and kept in memory; only fakeprints were saved (`features_hq\unseen\robust.npz`, 39 MB).
Each song was changed 11 ways, then scored with the saved cut-offs (nothing tuned):

| Change | Real flagged | Suno | MiniMax | Udio | Suno / MiniMax / Udio, joint rule |
|---|---|---|---|---|---|
| none (clean) | 8% | 100% | 88% | 95% | 100 / 95 / 95% |
| **speed −3% / −1% / +1% / +3%** (pitch and tempo both change) | 2–11% | **5–35%** | **0–10%** | **0–8%** | 83–93 / 88–95 / 0–3% |
| pitch −1 / +1 semitone (tempo kept) | 4–7% | 5–15% | 13–20% | 3–5% | 23–30 / 40–48 / 0–3% |
| tempo +5% (pitch kept) | **18%** | 95% | 55% | 68% | 93 / 50 / 48% |
| MP3 128 kbps | 10% | 100% | 68% | 88% | 100 / 95 / 80% |
| Ogg Vorbis ~145 kbps (Spotify's format) | 7% | 100% | 73% | 95% | 100 / 98 / 88% |
| EQ (bass +6 dB, treble −6 dB) | 7% | 100% | 88% | 95% | 100 / 98 / 90% |
| white noise, 30 dB below the song | 4% | 88% | 48% | 80% | 75 / 55 / 70% |

(Default model = `model_twoband.joblib`. 100 real songs, so ±3 points is noise. The per-family model behaves the same
way: speed ±1% drops it to 23–28% on Suno and 5–10% on Udio.)

**What this means:**
- **A 1% speed change breaks the classifier.** Suno falls from 100% to under 10%, Udio from 95% to 0–8%. Nobody hears a
  1% change, so this is the detector's biggest weakness. The reason: the classifier learned peaks at exact
  frequencies, and a speed change moves every peak a little.
- **The comb survives speed changes, and even measures them.** MiniMax's spacing moves from 689.07 Hz to 695.95 Hz at
  +1% (689.07 × 1.01 = 695.96) and 709.74 Hz at +3%. So the joint rule still catches Suno and MiniMax (83–95%).
  Udio has no comb, so nothing catches a sped-up Udio song.
- **Pitch shifting (with tempo kept) beats even the comb half the time**, because the phase vocoder also smears the
  peaks.
- **Time stretching makes real songs look AI** (18% flagged): the phase vocoder adds its own artifacts.
- **Normal re-encoding and EQ are fine.** MP3 and Ogg cost some MiniMax songs (88% → 68–73%), but the joint rule keeps
  95%. Added noise hurts more (MiniMax 48%).
- Fixes to try: train on sped-up and slowed-down copies (needs the training audio again), or make the fakeprint
  speed-proof: find the comb spacing first, then rescale the frequency axis before the classifier.

## Several windows per song instead of the first 30 s

```
python -I window_test.py      # 199 whole songs, 1.15 GB read, ~10 min; reports\window_test.json
```

`predict` only looks at the first 30 s. Here, whole songs never trained on (116 real: 79 Jamendo and 37 FMA tracks
read whole from `fma_full.zip`; 39 Suno V5.5, 38 MiniMax 3.0) were cut into 30 s windows (up to 8 = 4 min; median 6
per song), and each window was scored by the default model with its saved cut-off. Songs under 60 s were skipped.
No audio was kept; fakeprints are in `features_hq\unseen\windows.npz` (21 MB).

| Rule | Real flagged | Suno | MiniMax | AUC | Joint rule: real / Suno / MiniMax |
|---|---|---|---|---|---|
| First 30 s (what `predict` does now) | 4.3% | 100% | 84% | 0.974 | 4.3 / 100 / 92% |
| Middle window | 3.4% | 100% | 71% | 0.987 | 1.7 / 100 / 95% |
| Mean P(AI) over windows | 6.0% | 100% | **100%** | 0.997 | 0.9 / 100 / 95% |
| **Median P(AI) over windows** | **0.9%** | 100% | **97%** | **0.997** | 1.7 / 100 / 95% |
| One fakeprint of the whole song | 12.1% | 100% | 100% | 0.997 | 12.1 / 100 / 100% |

**What this means:**
- **The median over windows is clearly better:** fewer false alarms (4.3% → 0.9%, 5 songs → 1) and more MiniMax
  caught (84% → 97%). One odd window can no longer decide the verdict.
- **Any single window is noisy:** each position flags 2–6% of real songs, but 20% of real songs have at least one
  flagged window. So "AI if any window is AI" would be a bad rule.
- **The mean** is pulled up by single odd windows (6% false alarms). **One whole-song fakeprint** is smoother than the
  30 s fakeprints the model was trained on, so the saved cut-off no longer fits (12% false alarms), even though its
  AUC is just as high. It would need its own cut-off.
- Caveats: the cut-off was set for single windows, and 116 real songs is a small sample (0.9% is one song). The
  median needs at least two windows, so it only helps for files of 60 s or more.
- **`predict` now uses the median** for 44.1 kHz files of 60 s or more (up to 8 windows; the comb scores are the
  median too, and the exact spacing comes from the window where the comb stands out most). Shorter files still use
  their first 30 s. `--first-30s` gives the old behaviour. The model and the stored fakeprints are unchanged: each
  window is an ordinary 30 s fakeprint, like the training clips.

## Combining the two logistic regressions

```
python -I combo_test.py      # ~1 hour, no download; reports\combo_test.json
```

Per family has the fewest false alarms; one boundary does better on brand-new generators (MiniMax). Can a mix give
both? Both models were trained in the same splits on the same songs (as `nonlinear_test.py` with the varied real
songs, live concerts kept out). Each score becomes "share of real songs below it", and then the two are combined by
the higher one (**max**) or the average (**mean**). Each rule gets its own cut-off at 5% of real songs. Rules fixed
before running.

| | One boundary | Per family | Max | Mean |
|---|---|---|---|---|
| AI caught, cross-validated (17 generators) | 87.3% | **90.2%** | 89.3% | 89.5% |
| AI caught, family never seen | 64.9% | 64.4% | 64.7% | **66.4%** |
| Real flagged in that test | **5.0%** | 5.7% | 5.5% | 5.7% |
| New real songs flagged (fresh44) | 5.6% | **5.0%** | 5.3% | 5.3% |
| Live concerts flagged (never trained on) | 7.1% | **5.1%** | 6.0% | 6.5% |
| Suno V5.5 / MiniMax / Lyria 3 (fresh44) | 97 / **80** / **22%** | **99** / 71 / 18% | 99 / 76 / 16% | 98 / **80** / 18% |
| Mureka / Lyria Pro 3, cross-validated | 40 / 38% | **50 / 43%** | 44 / 38% | 47 / 44% |

**What this means:**
- **Mixing gives a middle ground, not both strengths.** The mean keeps one boundary's MiniMax score (80%) and is
  about as good as per family on known generators. But it also takes some of one boundary's false alarms
  (live concerts 6.5% vs 5.1%). Its gain on never-seen families is small (66.4% vs 64.9%), and that test flags a bit more real songs.
- So it is not worth a third model. **Per family alone is still the best all-round choice**, and on new comb
  generators the joint rule does more than any mix: adding the precise comb catches 94% of MiniMax with every rule.
  With the comb added, the cross-validated catch rate drops (82–85%), because the comb uses part of the 5% false-alarm budget.
- `predict` is unchanged.

## More information than the fakeprint? (shape, stereo, time)

```
python -I fpinfo_test.py                    # ~1 hour, no download; reports\fpinfo_test.json
python -I temporal_build.py --workers 8     # ~21 GB streamed, ~3 hours; only features kept (features_tm\)
python -I temporal_test.py                  # ~16 hours (without the "all" set); reports\temporal_test.json
python -I shapeaug_test.py                  # ~30 min; reports\shapeaug_test.json
```

The fakeprint keeps only the small peaks of the average spectrum. It throws away how strong they are, the
spectrum's overall shape, how the spectrum changes over time, and the stereo image. Does any of that help?

**First check, FMA small + Echoes only (`fpinfo_test.py`, 5–16 kHz band, logistic regression).** Keeping the peak
strength (no scaling, no clip) or the dips changed nothing (cross-validated 92.7–94.9%). Adding the overall shape
(0–22 kHz in 500 Hz bands, relative to 5–8 kHz) looked like a big win: cross-validated 94.8 → 97.5%, generator
never seen 72.2 → 83.9%. But all the real songs in that test were FMA small, so a feature that learns "looks like
an FMA song" would also look good there. So the full test below checks on other real songs.

**Full test (`temporal_build.py` + `temporal_test.py`).** All training and test songs were streamed again
(~21 GB, no audio kept) to build new features next to the fakeprint, from the same pass:
- **shape**: mean dB per 100 Hz, 0 Hz to Nyquist, each band relative to its 5–8 kHz level (301 values);
- **stereo**: side minus mid per 100 Hz up to 16 kHz, from the original file (160 values);
- **time**: how much each frequency bin moves over time (its spread over frames, minus a running median), and
  how the bins move together (11,856 values).

Six models × four feature sets (24 runs; an "all" set was stopped, since time alone already did badly). Songs as
`nonlinear_test.py` with the varied real songs; live concerts kept out to test on. All at 5% false alarms on real
training songs. Mean over the six models:

| | Fakeprint (now) | + shape | + stereo | + time |
|---|---|---|---|---|
| AI caught, cross-validated | 89.1% | **93.9%** | 91.9% | 86.8% |
| AI caught, family never seen | 63.8% | 64.2% | **67.9%** | 59.9% |
| Real flagged in that test | 5.2% | 5.4% | **4.9%** | 5.6% |
| **New real songs flagged (fresh44)** | **6.7%** | 9.7% | 8.8% | 13.5% |
| **Live concerts flagged** | **5.2%** | 14.8% | 6.9% | 5.6% |
| MiniMax / Lyria 3 / Suno V5.5 (fresh44) | **89** / 32 / **99%** | 67 / 51 / 99% | 71 / 66 / 98% | 58 / **73** / 90% |

The two best models, in detail (classifier alone; in brackets with the precise comb, one joint cut-off):

| | LR per family, now | + stereo | LightGBM, now | + stereo |
|---|---|---|---|---|
| AI caught, cross-validated | 90.3 (86)% | 94.4 (90)% | 92.2 (86)% | **95.9 (92)%** |
| AI caught, family never seen | 64.7 (58)% | 69.5 (62)% | 67.1 (59)% | **74.6 (68)%** |
| New real songs flagged (fresh44) | **5.0** (5.6)% | 6.1 (8.2)% | 8.0 (5.8)% | 10.6 (6.9)% |
| Live concerts flagged | 5.3 (3.8)% | 8.7 (4.9)% | 5.1 (2.2)% | 4.9 (3.6)% |
| MiniMax / Lyria 3 (fresh44) | 73 / 18 (94 / 6)% | 74 / 43 (94 / 10)% | **99** / 55 (97 / 31)% | 88 / **92** (95 / 75)% |

**What this means:**
- **Shape is a shortcut.** It wins on the training-style tests and loses on new real songs: live concerts flagged
  15% on average (per family: 26%) instead of 5%, and MiniMax drops. It mostly learns how FMA songs are mixed and
  recorded, not what AI audio looks like. So the gain in the first check was not real.
- **Random EQ does not fix the shape shortcut (`shapeaug_test.py`).** During training, each song's shape got a
  random smooth EQ curve (tilt, bass and treble shelves, optionally a low-pass), so mixing style would no longer
  separate real from AI. Test songs were not changed. Even with 6× stronger curves, live concerts were still
  flagged 16% (LR one boundary) and 18% (per family), against 17.6% and 25.6% with no EQ and about 5–7% without
  shape. So the shortcut is not a smooth EQ difference; it is finer detail in the shape.
- **Time is worse.** New real songs flagged 13.5% on average (every model 11–16%), fewer songs from never-seen
  families, and Suno V5.5 drops for some models. It does help Lyria 3, but at too high a cost.
- **Stereo is the only promising one.** It helps every model on families never seen (+4 points on average) and
  Lyria 3 a lot (LightGBM 55 → 92%), and live concerts rise only a little (6.9% on average). But new real songs are
  flagged more (8.8% vs 6.7% on average), and it only works on the original stereo file (a mono copy gives
  no stereo information). LightGBM + stereo + comb is the best candidate seen so far (92% cross-validated, 68% on
  never-seen families, 95% MiniMax, 75% Lyria 3) but flags 6.9% of fresh44's real songs, and it was chosen after
  looking at fresh44, so it needs a new set of real songs before it can replace anything.
- `predict` is unchanged: per-family LR on the fakeprint is still the safest model.

**Check on new real songs (`newreal_test.py`).** LightGBM + stereo + comb was then tested on 1,077 real songs at 44.1 kHz
or more, never used before: Internet Archive Jamendo albums (400), hip-hop mixtapes (289) and Community Audio music
(388). All were uploaded before 2022, so before AI song generators, and there is one song per artist. Download:
1.1 GB streamed, only features kept. Models trained on all training songs, cut-offs from cross-validation (5% of real
training songs). Rule fixed before running: it passes if it flags at most 6% of these songs, and at most 1 point
more than LR per family + comb.

| Real songs flagged: classifier (with comb) | LR per family | LR one boundary | LightGBM | LightGBM + stereo |
|---|---|---|---|---|
| Jamendo | 5.0 (6.0)% | 6.0 (7.2)% | **2.2** (5.5)% | 8.0 (6.5)% |
| Hip-hop mixtapes | **6.2** (6.9)% | 9.7 (7.6)% | 14.5 (10.4)% | 14.5 (8.3)% |
| Community Audio | **4.4** (6.2)% | 4.6 (6.2)% | 8.2 (7.7)% | 9.3 (8.5)% |
| **All new real songs** | **5.1 (6.3)%** | 6.5 (7.0)% | 7.7 (7.6)% | 10.2 (7.7)% |

- **LightGBM + stereo fails:** 7.7% of new real songs flagged with the comb (10.2% alone), against 6.3% for LR per family + comb.
  Hip-hop mixtapes are the worst (8–15% for LightGBM): a genre FMA small has few songs of.
- **LR per family holds up well:** 5.1% on brand-new real songs, right at the 5% target it was set for. So it stays
  the model in `predict`.
- The comb adds about one point of false alarms on these songs (5.1 → 6.3% for per family), a bit more than on the training songs.

## A fakeprint that ignores where the peaks sit (autocorrelation)

```
python -I autocorr_test.py      # ~1.5 hours, no download; reports\autocorr_test.json
```

The fakeprint records where each peak is; a never-seen generator puts its peaks elsewhere. The autocorrelation of the
fakeprint describes the pattern instead ("peaks every 86 Hz"). Lags up to 1000 Hz in both bands (1,396 values),
computed from the stored fakeprints. Same songs and tests as `temporal_test.py`, plus the 1,077 new real songs. Rule
fixed before running: fakeprint + autocorrelation (LR per family) must catch at least 2 points more of never-seen
families, flag at most 6% of the new real songs, and at most 1 point more on fresh44's real songs and the live concerts.

| Classifier (with comb) | LR per family, fakeprint | Autocorrelation only | Fakeprint + autocorrelation | LightGBM, autocorrelation only |
|---|---|---|---|---|
| AI caught, cross-validated | 90.3 (85)% | 61.8 (56)% | **93.1** (84)% | 89.9 (83)% |
| AI caught, family never seen | 64.7 (58)% | 43.2 (49)% | 68.3 (59)% | **69.7 (60)%** |
| New real songs flagged | **5.1** (6.3)% | 4.8 (6.3)% | 6.0 (6.4)% | 7.9 (8.1)% |
| fresh44 real / live concerts flagged | 5.0 / 5.3% | 4.8 / **1.6%** | 5.3 / 2.7% | 6.1 / 2.9% |
| MiniMax / Lyria 3 (fresh44) | 73 / 18% | 1 / 2% | **77 / 31%** | 47 / 47% |
| Mureka / Udio, never seen | 20 / 18% | 1 / 3% | 30 / **34%** | **62** / 28% |

**What this means:**
- **Just misses the rule.** Fakeprint + autocorrelation catches 3.6 points more of never-seen families, and live
  concerts are flagged half as often. But it flags 6.04% of the new real songs (65 of 1,077, against 55 for the
  fakeprint alone; mostly hip-hop mixtapes and Community Audio), just over the 6% limit. The difference is
  within noise (±0.7 points), but the rule was fixed before running, so the model is not changed.
- **With the comb, the gain almost disappears:** never-seen families 58 → 59%, new real songs 6.3 → 6.4%. The
  comb already catches most of what the autocorrelation adds.
- **Alone, the autocorrelation is weak for LR** (43% on never-seen families). It does better with LightGBM
  (69.7%, best of all on never-seen families, Mureka 62%), but LightGBM again flags too many new real songs (7.9%).
- `predict` is unchanged.

## Whole song instead of 30 s

```
python -I wholesong_test.py --build      # ~9 GB streamed, ~1.5 hours; features_tm\wholesong\ (no audio kept)
python -I wholesong_test.py              # ~30 min; reports\wholesong_test.json
```

Every fakeprint so far is the average spectrum of 30 s. Averaging the whole song (here up to 8 minutes; median
3–4 min) should give a cleaner fakeprint. The real training songs are mostly FMA 30 s clips, so training on whole
songs could teach "less noise = AI". So the models stay trained on 30 s, and only complete test songs are scored
both ways: the new real songs (1,051 complete ones) and fresh44's 79 Jamendo songs for false alarms, and Suno V5.5,
MiniMax and Lyria 3 (313 songs) for the catch rate. Same songs in both versions.

**With the 30 s cut-off, whole songs score higher across the board**, real songs too (LR per family):

| | 30 s | Whole song |
|---|---|---|
| Real songs flagged (1,130) | **4.8 (6.2)%** | 13.9 (14.6)% |
| AI songs caught (313) | 78.6 (81.2)% | **96.8 (92.3)%** |

A cleaner fakeprint makes every song's peaks stand out more, so the cut-off made on 30 s clips does not fit.
**A fair comparison gives each version its own cut-off** at 5% of real songs of its own kind. Half the complete
real songs set the cut-off, the other half measures false alarms (in brackets: with the comb):

| Own cut-off per version | LR per family, 30 s | LR per family, whole song | LR one boundary, 30 s | LR one boundary, whole song |
|---|---|---|---|---|
| Real songs flagged (other half, 565) | 7.1 (8.8)% | **5.7 (5.5)%** | 6.2 (6.2)% | 6.0 (5.7)% |
| Suno V5.5 | 99.3 (98.0)% | **100 (100)%** | 96.7 (96.7)% | **100 (100)%** |
| MiniMax | 83.0 (94.6)% | **99.1** (96.4)% | 78.6 (93.8)% | 98.2 (97.3)% |
| Lyria 3 | 21.6 (9.8)% | **56.9** (29.4)% | 13.7 (3.9)% | 49.0 (21.6)% |
| **All AI** | 80.8 (82.4)% | **92.7** (87.2)% | 76.7 (80.5)% | 91.1 (86.3)% |

**What this means:**
- **The whole song is much better at test time:** about 12 points more AI caught at the same false-alarm level.
  MiniMax goes to 99% without the comb, and Lyria 3 (the hardest generator so far) from 22% to 57%. It is the
  biggest gain of any idea tested here, and it needs no new training.
- **But it needs its own cut-off.** With the 30 s cut-off, it flags 14% of real songs. Here the cut-off was set on
  half of the test songs, so a version for `predict` needs a set of complete real songs to calibrate on, kept
  apart from any test set.
- The comb gains less from the whole song than the classifier does (MiniMax with comb 95 → 96%). For Lyria 3 the
  joint rule is worse than the classifier alone, as before.
- Caveats: fresh44's AI songs have been used before; 313 AI songs from three generators; Lyria 3 has only 51 songs.
- `predict` is unchanged for now (it uses the median over 30 s windows).

**Then in `predict`, checked on a new test set (`calibrate_whole.py`, `wholecheck_test.py`).** The whole-song cut-off
and real-song ranks were set on the 1,130 complete real songs above (cut-off P(AI) > 0.34 instead of 0.10). Then
both ways of scoring were checked on 1,200 songs never used before (8 GB streamed, no audio kept): 600 complete real
songs by new artists (Jamendo, Community Audio, netlabels; uploaded before 2022) and 600 unused Suno V5.5 and MiniMax
songs. Model: `model_twoband.joblib` (classifier; in brackets with the comb):

| Flagged | Windows (median of 30 s), 30 s cut-off | Whole song, whole-song cut-off |
|---|---|---|
| Jamendo (real, 200) | 3.0 (4.0)% | 7.5 (4.5)% |
| Community Audio (real, 200) | 3.0 (4.0)% | 5.5 (6.5)% |
| Netlabels (real, 200) | 4.0 (7.0)% | 8.0 (9.5)% |
| **All real (600)** | **3.3 (5.0)%** | 7.0 (6.8)% |
| Suno V5.5 (300) | 100 (100)% | 100 (100)% |
| MiniMax (300) | 91.7 (96.0)% | **99.3 (99.0)%** |
| **All AI (600)** | 95.8 (98.0)% | **99.7 (99.5)%** |
| Real songs above the 90 / 95 / 99% ranks (expected 10 / 5 / 1%) | 7.7 / 3.3 / 0.8% | 9.7 / 7.0 / 2.0% |

**What this means:**
- **Whole song catches more AI, but its cut-off does not hold.** It flags 7.0% of new real songs instead of 5%
  (netlabels 8%: a source not among the calibration songs). The window method stays under 5% here (3.3%).
- So the gain in `wholesong_test.py` was partly paid for with false alarms: the cut-off there was set on the same
  kind of songs it was tested on. 1,130 calibration songs from three sources are not varied enough.
- Both AI sets here are easy (Suno is caught 100% either way), so this test cannot show the gain on hard
  generators like Lyria 3.
- To use the whole song safely, the cut-off needs more varied complete real songs (more sources, genres and
  recording types), then a new check.
- **Decision:** `predict`, the web demo and the Colab demo went back to **windows as the default** (fewer false
  alarms matter more than a few more AI songs caught, when the answer is about real people's music). The whole song
  stays as an option: `detect.py predict song.mp3 --whole`, or `demo.show(path, mode="whole")`.

## Generators it never saw: stretching, and would more songs help? (stretch_test.py, learning_test.py)

The weak spot is generators the model never trained on (about two thirds caught). Two questions, no download:

**1. Stretch the fakeprints while training (`stretch_test.py`).** Half of the training songs get both fakeprints
stretched along the frequency axis by a random factor (up to ±3% or ±10%), like a small speed change. The hope:
the model learns "evenly spaced peaks" instead of "peaks at these exact frequencies". Quick run on 3,000 songs
(the full run was not needed):

| LR per family | No stretch | ±3% | ±10% |
|---|---|---|---|
| AI caught, 5-fold cv | 88.3% | 78.6% | 72.7% |
| AI caught, family never seen | **69.6%** | 59.7% | 54.5% |
| MiniMax / Suno V5.5 (fresh44) | 88 / 99% | 51 / 95% | 32 / 87% |
| Speed −1% / +1% (robustness songs, AI caught) | 18 / 15% | 23 / 14% | 14 / 25% |

LR one boundary showed the same: 50.8% → 36.9% → 27.9% of never-seen families. **Stretching makes everything
worse and does not fix speed changes.** A linear model on the fakeprint cannot learn "a comb anywhere": it can only
spread its weights thinner. Position-free patterns need a different feature (the precise comb does this; the
autocorrelation was a near miss).

**2. Learning curve (`learning_test.py`, LR one boundary).** Is a generator weak because there are too few of its
songs, or because its trace is faint? Each weak family is split into 3 folds by group; the model trains on 0, 1/3,
2/3 or all of the other two folds (cut-off: 95% of real songs held out by artist):

| Family | 0 songs | 1/3 | 2/3 | all (songs) | Last step |
|---|---|---|---|---|---|
| Udio (control) | 8.0% | 82.6% | 88.1% | 89.9% (867) | +1.8: levelled off |
| Sonauto | 32.4% | 49.1% | 63.2% | 67.3% (333) | +4.2: still rising |
| Mureka | 20.5% | 29.2% | 33.5% | 37.5% (441) | +3.9: still rising |
| Lyria | 22.0% | 25.0% | 30.0% | 33.0% (67) | +3.0: still rising |
| Mubert | 6.0% | 9.4% | 12.1% | 15.4% (99) | +3.4: still rising |

**What this means:**
- **Seeing a generator's songs is what works.** Udio goes from 8% (never seen) to 83% with about 400 of its songs.
  So the practical answer to new generators is to keep adding their songs and retrain.
- **Mureka, Lyria and Mubert have a faint trace:** even with all their songs they stay under 40%, rising only
  3–4 points per step. More songs would help a little, but not enough.
- **Hardly any more songs exist:** all 662 Mureka songs (MUSIC8K) and all 100 Lyria Pro 3 songs (HAIM) are already
  used; Lyria 3 has 51 (sleeping-ai/Lyria3); no Mubert dataset was found. Only Sonauto has many more
  (ai-music-deduplicated: 26 GB, 500 songs used so far).

### Is the faint generators' trace somewhere else in the spectrum? (fullband_probe.py)

The first 60 s of 50 Mureka, 51 Lyria 3, 20 Suno V5.5 (a known comb, as a check) and 50 real Jamendo songs (252 MB
streamed, nothing saved). Fakeprint and precise comb in four bands. "Beat real" = share of songs whose precise comb
is above 95% of the real songs (5% by chance). "Mean-fp comb" = the precise comb of the generator's average
fakeprint: peaks at the same place in every song add up there (real songs: about 4.4).

| | 0.5–2 kHz | 2–5 kHz | 5–16 kHz (used now) | 16–20 kHz | Top of content |
|---|---|---|---|---|---|
| Real | 6% · 4.5 | 6% · 4.3 | 6% · 4.3 | 6% · 4.5 | 18.4 kHz |
| Suno V5.5 (check) | 10% · 5.0 | **90% · 15.6 (200 Hz)** | **90% · 17.3 (200 Hz)** | 25% · 12.1 (200 Hz) | 17.3 kHz |
| Mureka | 10% · 4.9 | 4% · 5.8 | 4% · **10.1 (300 Hz)** | 6% · 5.8 | 18.5 kHz |
| Lyria 3 | 8% · 3.8 | 2% · 6.0 | 0% · 5.1 | 0% · 2.2 | 15.8 kHz |

**What this means:**
- **No other band helps.** The check works (Suno's 200 Hz comb shows up in three bands), but neither Mureka nor
  Lyria has more of a trace below 5 kHz or above 16 kHz.
- **Mureka has a comb, at exactly 300 Hz in 5–16 kHz, but it is too faint to see in one song** (4% of songs beat
  the real songs); it only appears when 50 songs are averaged. That fits "37% caught even when trained on". A check
  that looks only at that one known spacing, or a longer average (whole song), may find it in single songs.
- **Lyria 3 shows nothing anywhere** (and its files stop at about 16 kHz). Its audio is probably made in a way that
  leaves no upsampling comb, so the fakeprint method cannot catch it; it needs a different kind of detector.

### A comb check aimed at known spacings (known_spacing_test.py)

The precise comb searches every spacing from 20 to 2000 Hz and keeps the best, so real songs also get a high "best"
by chance. If a generator's spacing is known, checking only that spacing is fairer. For each of the 16 AI families,
the spacing comes from the mean fakeprint of half its songs (by group); the other half is tested. The 13 families
with a clear comb in that mean form a **bank** of known spacings (e.g. Suno, Brev and ElevenLabs all at 200 Hz;
ACE-Step and Stable Audio at 172.27 Hz; Udio 23.44 Hz; Mureka 600 Hz). Lyria, Mubert and Producer have no clear comb.
"bank" = the song's best known-spacing score over all 13 spacings. Stored 30 s fakeprints, no download. All at 5% of
real training songs flagged:

| AI caught (test halves) | Known spacing (own) | Precise comb (now) | Bank | Classifier (LR) | Classifier + precise | **Classifier + bank** |
|---|---|---|---|---|---|---|
| Mureka | **27.5%** | 2.4% | 28.4% | 35.3% | 26.6% | 32.9% |
| Udio | 75.7% | 0.1% | 63.9% | 93.8% | 90.2% | 91.5% |
| Sonauto | 73.6% | 16.0% | 64.0% | 77.2% | 66.0% | 76.8% |
| ElevenLabs | 97.3% | 62.0% | 90.7% | 82.7% | 82.0% | **94.0%** |
| AudioLDM | 99.7% | 20.4% | 92.9% | 100% | 99.3% | 99.7% |
| Lyria / Mubert | – | 2 / 1% | 2 / 3% | 42 / 12% | 30 / 11% | 28 / 12% |
| **All AI** | – | 35.5% | **75.6%** | 87.5% | 84.6% | **87.2%** |
| **Real songs never used, flagged** (1,903: fresh44, 1,077 new, live) | – | 5.4% | 4.9% | 6.5% | 6.4% | **5.5%** |

Both rules fixed before running pass: known spacing catches 25 points more Mureka than the precise comb (rule: +15),
and classifier + bank catches 2.6 points more AI than classifier + precise comb with 5.5% of never-used real songs
flagged (rule: at most 6%).

**What this means:**
- **Knowing the spacing makes the comb far stronger:** the bank alone catches 75.6% of AI songs with no classifier
  at all (the precise comb: 35.5%), with fewer false alarms (4.9%).
- **Classifier + bank beats today's joint rule** (+2.6 points of AI) **and flags fewer real songs** (5.5% vs 6.4%,
  even fewer than the classifier alone). The bank also says which spacing matched, i.e. which kind of decoder.
- **Mureka's faint comb is now found in about 1 song in 4** (2.4% → 27.5%), but the classifier is still a bit better
  on Mureka. Lyria and Mubert stay out of reach: no comb.
- **Limits:** the bank only knows spacings it has seen (a new generator is caught only if it shares a decoder rate
  with a known one), and an exact spacing moves with a speed change, where the precise comb survives. Not yet
  checked on fresh44's AI songs or the robustness songs.

## Status and next steps (Nov 24, 2026)

- Bump removal (`remove_bumps()`, `comb_score(..., bump_hz=15)`) and the precise comb (`precise_comb()`) are in
  `fakeprint.py`. `detect.py predict` uses them and prints the exact spacing. The `selftest` and `train` reports and
  `scan_sonics.py` still use the old comb score.
- Echoes has now been used to test two versions of the precise comb, so it is no longer fully unseen for this
  method. A future change should be checked on a new set of generators.
- The fresh songs have now been used for five tests (bump removal, more data, cut-off, artist check, and as
  the real songs in the unseen-generator test). Nothing was tuned on them, but a new idea that needs tuning
  should get a new, unused test sample.
- `detect.py predict` uses the two-band model (`model_twoband.joblib`, v1 retrained with the varied real songs)
  for 44.1 kHz files (median over 30 s windows for songs of 60 s or more) and `model_family` for the rest;
  `--joint` adds the precise comb with one joint cut-off. It was tested on 728 new 44.1 kHz songs (`fresh44_test.py`). That set has now been used twice (two-band v1, then v2): a new
  idea that needs tuning should be tuned elsewhere and then checked on it. (It has since also checked v2 and the
  per-family model.)
- Next ideas:
  1. Six models were compared on the same fakeprints (`nonlinear_test.py`), with 1,500 more varied real songs
     (`more_real.py`). Per-family logistic regression is still the best all-round model. Mureka and Lyria still
     need more songs (Lyria Pro 3: only 100 exist so far), or a feature that sees more than the average spectrum.
     No new Mubert songs were found.
  2. Done: robustness test (`robust_test.py`). A 1% speed change breaks the classifier; the comb survives and
     measures the change. Next: a speed-proof classifier (speed-changed training copies, or rescaling by the comb).
  3. Use the precise comb in the `selftest` and `train` reports too.
  4. Done: combining the two logistic regressions (`combo_test.py`): a middle ground, not worth a third model.
  5. Done: more information than the fakeprint (`temporal_test.py`). Shape and time features are shortcuts (more
     false alarms on new real songs). Stereo helps on new generators but still raises false alarms a little.
     LightGBM + stereo + comb then failed on 1,077 never-used real songs (`newreal_test.py`: 7.7% flagged, vs 6.3%
     for LR per family + comb), so stereo is not used. Those songs have now been used once too.
  7. Done: autocorrelation of the fakeprint (`autocorr_test.py`): +3.6 points on never-seen families, but just over
     the false-alarm limit on the new real songs (6.04% vs 6%), and almost no gain with the comb. Not used.
  8. Done: whole song instead of 30 s at test time (`wholesong_test.py`): with its own cut-off, 92.7% of new AI songs
     caught vs 80.8%, at about 5–6% false alarms. Then in `predict` with its own cut-off (`calibrate_whole.py`),
     checked on 1,200 new songs (`wholecheck_test.py`): 99.7% of AI caught vs 95.8% for windows, but 7.0% of real
     songs flagged vs 3.3%. So windows stay the default and the whole song is an option (`--whole`).
     Next: calibrate the whole song on more varied complete real songs, then check it again on new songs.
  9. Done: stretch augmentation (`stretch_test.py`): worse everywhere, not used. Learning curve
     (`learning_test.py`): new generators need their own songs; Mureka, Lyria and Mubert stay weak even then.
     Full-band check (`fullband_probe.py`): no other band helps; Mureka has a faint 300 Hz comb, Lyria 3 none.
     Known-spacing bank (`known_spacing_test.py`): classifier + bank catches 87.2% vs 84.6% (classifier + precise
     comb) with fewer false alarms (5.5% vs 6.4%). Next: check it on fresh44's AI and the robustness songs, then
     use it in `predict --joint`.
  6. fresh44 and the live concerts have now also been used to judge the new features, so a model chosen from
     these results needs a new, unused test set.

## Quick start (Windows, from this folder)

```
python detect.py selftest
```
This opens nothing by itself. Open `reports\selftest.html` in your browser, and
listen to the `reports\example_*.wav` files.

Install dependencies in a new environment: `pip install -r requirements.txt`

## Web demo (Streamlit)

```
python -m streamlit run app.py      # then open http://localhost:8501
```

A one-page website for people who don't code: upload a song (MP3, WAV, FLAC, OGG) and it says **Likely made by
AI**, **Unsure** or **Likely made by people**, with how the song ranks among human-made songs, the model's P(AI), and
one chart: the song's fingerprint (5–16 kHz), with dashed lines when the peaks repeat at an exact spacing. It uses
the same steps as `detect.py predict` (median over 30 s windows for 44.1 kHz files; the 16 kHz model for lower
rates), through `demo.analyze` and the plain-number models in `demo_models\`. Uploaded files are not stored.

**Free hosting** (needs an account that you create yourself):
- *Streamlit Community Cloud*: put the project on GitHub, sign in at share.streamlit.io with GitHub, choose
  "Create app", pick the repository and `app.py`. It installs `requirements.txt` by itself.
- *Hugging Face Spaces*: create a Space with the Streamlit template, then upload `app.py`, `demo.py`, `detect.py`,
  `fakeprint.py`, `audio_io.py`, `fetch_data.py`, `synth.py`, `report.py`, `requirements.txt` and the `demo_models`
  folder (about 1 MB in all).
Either way you get a public link to send. Only the code and the 1 MB of models go online, no training data.

## Demo (Google Colab)

Open `demo.ipynb` in Colab (File → Upload notebook), run the cells, and upload `dist\ai-music-detector-demo.zip`
when asked. The zip (0.6 MB) holds the code and the two models as plain numbers (weights, cut-offs, real-song scores;
no pickles), so any scikit-learn version works. The notebook first scores two synthetic clips, then any songs you
upload. Each song gets `predict`'s answer and a plot: its rank among real songs (likely real / unsure / likely AI)
and the fakeprint of both bands, with the comb marked when there is one. Like `predict`, it uses the median over
30 s windows; `demo.show(path, mode="whole")` scores the whole song instead. After retraining, rebuild the zip:

```
python -I demo.py      # demo_models\ from model_twoband.joblib + model_family.joblib, then dist\ai-music-detector-demo.zip
```

## Data: downloaded when needed, not stored

Big datasets aren't kept on disk. `fetch_data.py` downloads a dataset, keeps a small
sample, and deletes the zip. Run it again whenever you want more or different songs.

```
python -I fetch_data.py sonics --n 100
python -I fetch_data.py fma --n 100
```
- `sonics`: 50 Suno + 50 Udio songs into `data\ai\sonics\` (about 70 MB). Deletes the 2.7 GB zip.
- `fma`: 100 real Creative Commons clips from the Free Music Archive into `data\real\fma16k\` (about 15 MB),
  converted to the SONICS format (16 kHz mono, 40 kbps MP3) so the file format isn't a giveaway.
  Checks the download's SHA1, then deletes the 7.2 GB zip.
- `--seed 1` picks a different random sample, `--part 3` downloads another SONICS part (1–10),
  `--add` adds to the folder instead of replacing it, and `--keep-zip` keeps the download.

For training on thousands of songs, `build_fakeprints.py` keeps only the fakeprints, not the audio
(see "More training data" above). Its output goes to `features\`, except Echoes, which goes to
`features\unseen\` so it never joins the training pool. The 44.1 kHz versions (5–16 kHz spectra) are in
`features_hq\`. `fetch_data.py fma-artists` keeps only
a small track → artist table from the FMA metadata (`features\fma_artists.csv`).

## Run it on real songs

1. Put human-made songs in `data\real\` and AI songs in `data\ai\` (10+ each to start, more is better).
2. Train and evaluate:
   ```
   python detect.py train
   ```
   This writes `reports\train.html` (cross-validated accuracy, charts) and `model.joblib`.
3. Score any file:
   ```
   python detect.py predict path\to\song.mp3 --match-format
   ```
   Files at 44.1 kHz or more get the two-band model (`model_twoband.joblib`, `train_twoband.py`), which always
   converts them like the training songs. Songs of 60 s or more are scored as the median over 30 s windows (up to
   8; see "Several windows per song"); `--first-30s` scores only the start. Other files get the first model that exists: `model_family.joblib`
   (`family_test.py --phase save`), `model_final.joblib` (`train_final.py`), or `model.joblib` (`train`).
   For those, `--match-format` first converts the song to 16 kHz, 40 kbps MP3, like every song in the tests.
   Example output (two synthetic 44.1 kHz clips from `synth.py`, and fresh 16 kHz songs never trained on):
   ```
   (44.1 kHz files: model_twoband.joblib, both bands, AI if P(AI) > 0.10; median over 30 s windows, up to 8 (first 30 s if the song is under 60 s))
   (other files: model_family.joblib, AI if P(AI) > 0.54)
   synth_ai.wav [44.1 kHz, both bands]: precise=11.9 / 14.1 -> peaks repeat every 137.813 Hz  P(AI)=0.65 -> AI
   synth_real.wav [44.1 kHz, both bands]: precise=4.2 / 4.2 (no exact comb)  P(AI)=0.00 -> real
   fake_39710_suno_0.mp3: comb=0.26 precise=12.2 -> peaks repeat every 200.002 Hz (like Suno / Brev / ElevenLabs)  P(AI)=1.00 -> AI, most like suno
   fake_39621_udio_0.mp3: comb=0.08 precise=4.4 (no exact comb)  P(AI)=1.00 -> AI, most like udio
   fma_000255.mp3: comb=0.07 precise=3.8 (no exact comb)  P(AI)=0.00 -> real
   ```
   With both bands, `precise` shows the 1–8 kHz and 5–16 kHz scores. An exact comb is reported when a precise
   comb is above its cut-off (9.32 for 1–8 kHz, 10.35 for 5–16 kHz; 5% of real training songs go above that).
   The synthetic AI clip's comb (44,100 / 320 = 137.8 Hz) comes from its random decoder, so no known model is named.

   Add `--joint` to screen for new generators (44.1 kHz files only). The verdict then comes from the classifier
   **or** the precise comb, with one joint cut-off that still flags 5% of real training songs. It catches more new
   generators that have a comb (MiniMax 86% → 94%), but fewer of those without one (Udio 89% → 73%):
   ```
   synth_ai.wav [44.1 kHz, both bands]: precise=11.9 / 14.1 -> peaks repeat every 137.813 Hz  P(AI)=0.65 joint=0.994 -> AI
   ```
   `joint` is the song's highest score as the share of real training songs below it, and the cut-off is 0.982.
   "(by comb)" after the verdict means the classifier alone would have said real.

   **A rank instead of AI / real (44.1 kHz files, since Nov 23, 2026).** `P(AI)` is not a real probability: the
   model was trained as if half of all songs were AI. So `predict` now says how the song ranks among 9,378 real
   songs it was not trained on (cross-validated scores), with three levels: **likely real** (below 90%), **unsure**
   (90–99%), **likely AI** (above 99%). The old AI / real cut-off is 95%. These ranks hold on real songs never used
   before: the 90 / 95 / 99% levels flag 11 / 5.6 / 0.9% of the 1,077 new real songs (`newreal_test.py`),
   against 10 / 5 / 1% expected (with `--joint`: 12.5 / 6.9 / 1.2%). `--base-rate 0.1` also prints a rough chance
   that the song is AI if 10% of songs are AI. It compares how often AI and real training songs land in the same
   score range, so it is too low for new generators, which score lower than the training ones:
   ```
   synth_real.wav [...]: precise=3.9 / 3.7 (no exact comb)  P(AI)=0.00 -> likely real: higher than 38.5% of real songs; if 10% of songs are AI: <1% chance it is AI (rough)
   synth_ai.wav [...]: precise=10.8 / 13.0 -> peaks repeat every 137.811 Hz  P(AI)=0.29 -> unsure: higher than 97.8% of real songs (about 1 in 44 real songs scores this high); if 10% of songs are AI: ~14% chance it is AI (rough)
   ```
   Why not simply a probability: it depends on how common AI songs are. If 1 song in 100 is AI, then even at
   5% false alarms most flagged songs are real. The rank does not depend on that. `model_twoband.joblib` was
   retrained to store these scores (same model: identical weights and cut-offs). The 16 kHz models still print AI / real.

**Python:** `python` is any environment with `pip install -r requirements.txt` (Python 3.11 or newer).

**File formats:** WAV, MP3, FLAC and OGG (MP3/FLAC/OGG via `soundfile`).
In a quick test, the synthetic AI artifact survived MP3 compression: comb score 0.54 vs 0.09 for real.

**Where to get songs (use only audio you have the right to use, for research):**
- AI, newest models: your own Suno account. Since Sept 3, 2026 Suno caps downloads
  (free: up to 7 trial downloads in total; Pro: 20 a month). Udio downloads are switched off.
- AI, older models (free): the SONICS dataset (Suno v2/v3/v3.5 and Udio, CC BY-NC), via `fetch_data.py sonics`.
- Real: songs you own, or Creative Commons music from the Free Music Archive (FMA) or MTG-Jamendo.

**Important: keep both groups in the same format.** Use the same file type and
bitrate for real and AI songs, e.g. convert everything to 192 kbps MP3. Otherwise
the model can learn "MP3 vs WAV" instead of "AI vs real".

## Files

| File | What it does |
|---|---|
| `fakeprint.py` | Average spectrum, lower envelope, fakeprint, bump removal, comb score, precise comb |
| `synth.py` | Synthetic "real" clips and a random transposed-conv decoder for "AI" clips |
| `audio_io.py` | Loads WAV (and MP3/FLAC with soundfile) from a file or from memory, and resamples it |
| `report.py` | Interactive HTML report (Plotly, works offline) |
| `detect.py` | Command line: `selftest`, `train`, `predict` (two-band model for 44.1 kHz files, median over 30 s windows, ranked among real songs as likely real / unsure / likely AI; `--base-rate` adds a rough chance of AI; `model_family` for others; `--joint` adds the comb; `--whole` scores the whole song with its own cut-off) |
| `scan_sonics.py` | Training-free comb scan of SONICS songs, grouped by Suno/Udio, with HTML report |
| `fetch_data.py` | Downloads SONICS or FMA (or the FMA artist table), keeps a small sample, deletes the zip |
| `evaluate.py` | Held-out test: settings and classifier from `data\`, scored on `data_fresh\` |
| `build_fakeprints.py` | Downloads SONICS, FMA or Echoes, keeps only the fakeprints (`features\`), deletes the zip |
| `scale_test.py` | Trains on more and more of `features\`, tests each model on `data_fresh\` |
| `encoder_check.py` | Control: re-encodes the fresh songs to check the model isn't spotting the MP3 encoder |
| `cutoff_test.py` | Picks the P(AI) cut-off by cross-validation on `features\`, tests it on `data_fresh\` (`--artist-disjoint`: no shared artists) |
| `unseen_test.py` | Classifier, comb and precise comb on 12 generators never trained on (Echoes), per generator |
| `logo_test.py` | Leave one generator out: train on the other 11 (+ SONICS), test on the one left out |
| `train_final.py` | Final classifier on all 16k songs, cut-off by cross-validation → `model_final.joblib` |
| `family_test.py` | Same, with one boundary per generator family: tests, and → `model_family.joblib` |
| `hq_test.py` | 16 kHz (1–8 kHz) vs 44.1 kHz (5–16 kHz) on the same songs, with each source's bandwidth |
| `twoband_test.py` | Both bands together: classifier, precise comb, and both, with one joint 5% cut-off |
| `train_twoband.py` | Two-band classifier, cut-off by cross-validation (and the joint cut-off) → `model_twoband.joblib`; `--extra-dir` adds songs from more44 (→ v2) or more_real (→ the default model), `--family` one boundary per family |
| `fresh44_test.py` | Test on 728 new 44.1 kHz songs (real FMA and Jamendo, Suno V5.5, MiniMax, Lyria 3), reading only each song's start |
| `nonlinear_test.py` | Six models on the two-band fakeprints (LR one boundary, LR per family, LightGBM, neural network, random forest, RBF SVM): cross-validation, leave one family out, fresh44; `--extra-real` / `--hold-real` add or hold out real songs → `reports\nonlinear_*.json` |
| `window_test.py` | Whole songs cut into 30 s windows: first 30 s vs middle, mean, median and whole-song rules → `reports\window_test.json` |
| `combo_test.py` | One-boundary and per-family LR trained in the same splits, combined (max / mean of their percentiles, with and without the comb) → `reports\combo_test.json` |
| `fpinfo_test.py` | Does the fakeprint throw away useful information? Variants of the 5–16 kHz feature (no scaling, no clip, peaks + dips, + shape) from the stored spectra → `reports\fpinfo_test.json` |
| `temporal_build.py` | Streams the training and test songs again and keeps new features next to the fakeprints: shape, stereo width and time features → `features_tm\` (no audio kept) |
| `temporal_test.py` | Six models × feature sets (fakeprint, + shape, + stereo, + time, all): cross-validation, leave one family out, fresh44 and live concerts → `reports\temporal_test.json` |
| `shapeaug_test.py` | Random EQ on the training songs' shape feature, to see if it stops the shape shortcut (`--strength` for stronger curves) → `reports\shapeaug_test.json` |
| `newreal_test.py` | 1,089 never-used real songs (Internet Archive Jamendo albums, hip-hop mixtapes, Community Audio; uploaded before 2022), features only (`--build` → `features_tm\newreal\`); false alarms of four models → `reports\newreal_test.json` |
| `autocorr_test.py` | Autocorrelation of the fakeprint (ignores where the peaks sit), alone and added to it, LR and LightGBM; same tests plus the new real songs → `reports\autocorr_test.json` |
| `wholesong_test.py` | Whole song (up to 8 min) vs 30 s at test time, models trained on 30 s; with the 30 s cut-off and with each version's own cut-off (`--build`: full songs streamed → `features_tm\wholesong\`) → `reports\wholesong_test.json` |
| `app.py` | Web demo (Streamlit): upload a song → likely AI / unsure / likely made by people, P(AI), rank among human-made songs, fingerprint chart |
| `demo.py`, `demo.ipynb` | Colab demo: score one song, plot its rank among real songs and its fakeprint; `python -I demo.py` exports the models as plain numbers (`demo_models\`) and builds `dist\ai-music-detector-demo.zip` |
| `calibrate_whole.py` | Whole-song cut-off and real-song ranks for `predict`, from 1,130 complete real songs → `model_twoband.joblib["whole"]` |
| `wholecheck_test.py` | Checks whole-song `predict` on a new test set (new real songs, unused Suno V5.5 / MiniMax songs; `--build` streams them) → `reports\wholecheck_test.json` |
| `stretch_test.py` | Frequency-stretch augmentation of the training fakeprints (±3%, ±10%) vs none → `reports\stretch_test.json` (full run) |
| `learning_test.py` | Learning curve for the weak generators: share caught vs how many of their songs are trained on → `reports\learning_test.json` |
| `fullband_probe.py` | Mureka / Lyria 3 / Suno / real songs: fakeprint and precise comb in four bands (0.5–20 kHz), 60 s each streamed → `reports/fullband_probe.json` |
| `known_spacing_test.py` | Comb check at known spacings (a bank learned from half of each family's songs) vs the precise comb, alone and with the classifier → `reports/known_spacing_test.json` |
| `robust_test.py` | Robustness: 220 songs changed 11 ways (speed, pitch, tempo, MP3/Ogg re-encode, EQ, noise), scored by every model → `reports\robust_test.json` (`--rescore` re-scores saved fakeprints) |
| `more_real.py` | 1,500 more varied real songs (FMA large, Internet Archive netlabels and Live Music Archive), fakeprints only → `features_hq\more_real\` |
| `more44_test.py` | 2,762 more AI songs (Udio, Mureka, HeartMuLa, Sonauto, Lyria Pro 3) read from tar, parquet and single files; before/after tests (`--family`: per-family classifier) |

## Research ideas to take further

1. **Real-world robustness:** run real Suno/Udio songs through pitch shift, time stretch,
   MP3 re-encoding, and EQ. Does the comb score hold up better than the classifier?
2. **Spacing as a model fingerprint:** do different Suno/Udio versions show different peak
   spacings? Can you tell *which* model made a song?
3. **Combine both:** train a classifier on features that don't depend on where the peaks sit
   (e.g. the fakeprint's autocorrelation) so it generalises to new model versions.
4. **Attacks:** can a simple filter remove the comb without hurting the sound? This matters
   for anyone building a "verified human-made" music platform.

## Credit

The method and parameters follow the paper and the authors' released code
([deezer/ismir25-ai-music-detector](https://github.com/deezer/ismir25-ai-music-detector),
CC BY-NC 4.0, research use only). This is an independent re-implementation for
learning and research. The comb score, synthetic decoder and self-test are additions.

Data: SONICS (CC BY-NC), the Free Music Archive (Creative Commons tracks), and the Echoes dataset
([arXiv:2603.23667](https://arxiv.org/abs/2603.23667), CC BY-SA 4.0). Only fakeprints and small samples are kept.
The 44.1 kHz test (`fresh44_test.py`) also uses JamendoLyrics / jam-alt (Creative Commons songs from Jamendo),
the [Suno AI Music Dataset](https://huggingface.co/datasets/Kukedlc/suno-ai-music-dataset) (Suno V5.5, CC BY 4.0),
the [MiniMax Music 3.0 Dataset](https://huggingface.co/datasets/AngelSoftware/minimax-music-3-dataset) (CC BY 4.0)
and [sleeping-ai/Lyria3](https://huggingface.co/datasets/sleeping-ai/Lyria3) (no description; marked MIT).
Only the start of each song is read, and only fakeprints are kept.
`more44_test.py` uses [blanchon/udio_dataset](https://huggingface.co/datasets/blanchon/udio_dataset) (Udio, CC0),
[MUSIC8K](https://huggingface.co/datasets/homura23/MUSIC8K) (Mureka v9 and HeartMuLa, CC BY 4.0),
[ai-music/ai-music-deduplicated](https://huggingface.co/datasets/ai-music/ai-music-deduplicated) (Sonauto, MIT)
and [HAIM](https://huggingface.co/datasets/mippia/HAIM) (Lyria Pro 3, CC BY-NC 4.0;
[arXiv:2606.01686](https://arxiv.org/abs/2606.01686)), in the same way.
`more_real.py` uses FMA large and the Internet Archive's [netlabels](https://archive.org/details/netlabels)
(Creative Commons) and [Live Music Archive](https://archive.org/details/etree) (trade-friendly live recordings)
collections: 30 s of each song is read, and only fakeprints are kept.

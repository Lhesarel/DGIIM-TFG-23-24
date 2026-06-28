# Fourier Analysis Applied to Audio — Bachelor's Thesis (TFG)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![WAV→Score demo](https://img.shields.io/badge/🤗%20Demo-WAV→Score-orange.svg)](https://huggingface.co/spaces/JavierGraLo/wav-to-score)
[![2MANYFILTERS demo](https://img.shields.io/badge/🤗%20Demo-2MANYFILTERS-orange.svg)](https://huggingface.co/spaces/JavierGraLo/2manyfilters)

> **EN —** Two audio tools built on **Fourier analysis**: **WAV2MIDIConverter**, which transcribes a
> **monophonic** WAV recording into a musical score (MIDI + PDF) using an **original F0-detection
> algorithm** (FFT spectral analysis + dynamic K-Means clustering), and **2MANYFILTERS**, a
> frequency-domain audio filter (FFT → modify spectrum → IFFT). Built at the intersection of
> **mathematics, computer science and music**.
>
> **ES —** Trabajo Fin de Grado del doble Grado en Ingeniería Informática y Matemáticas (Universidad
> de Granada). Incluye un conversor de audio WAV monofónico a partitura (algoritmo propio de detección
> de frecuencia fundamental: FFT + K-Means dinámico) y una app de filtrado de audio en el dominio de la
> frecuencia. Combina métodos numéricos (análisis de Fourier) con conocimiento musical formal.

**Thesis:** *"Análisis de Fourier y aplicación práctica en el tratamiento de señales de audio"* ·
Double Degree in Computer Engineering & Mathematics, University of Granada (2023–2024) ·
Advisors: José Luis Gómez Ruiz (Mathematical Analysis) and Diego Salas González (Signal Theory).
The full thesis is included as [`TFG-JavierGranadosLopez-23-24.pdf`](TFG-JavierGranadosLopez-23-24.pdf).

---

## 🎬 Live demos (no install)

Interactive web versions, deployed on Hugging Face Spaces (the original algorithms unchanged; the
Tkinter GUI replaced by Gradio so they run in the browser):

| Demo | What it does | Link |
|------|--------------|------|
| **WAV → Score** | Upload a monophonic `.wav` → transcribed score (SVG) + MIDI + MusicXML | https://huggingface.co/spaces/JavierGraLo/wav-to-score |
| **2MANYFILTERS** | Upload a `.wav` → apply a low/high/band-pass/band-stop filter + before/after spectrum | https://huggingface.co/spaces/JavierGraLo/2manyfilters |

---

## Motivation

Reading music from a recording is something trained musicians do by ear, but automating it cleanly is
hard: it sits exactly where **mathematics**, **computer science** and **music** meet. This was my
Bachelor's Thesis for the **double degree in Computer Engineering and Mathematics (Universidad de
Granada)**, written by a formally trained composer (*Enseñanzas artísticas superiores de Composición*)
and professional viola player who is also a computer scientist and mathematician. That dual background
motivated an **original methodology for pitch recognition** rather than an off-the-shelf approach.

---

## 1. WAV2MIDIConverter — `WAV2MIDICONVERTER/`

Transcribes a **monophonic** WAV melody into a readable score.

**Pipeline (`audio → score`):**
1. **Audio input** — the `.wav` is read into a sample vector with `scipy.io.wavfile`.
2. **Windowing & spectral analysis** — the signal is split into **overlapping Hanning windows**, each
   transformed with the **FFT** (`numpy.fft`); the FFT magnitude is used as the amplitude spectrum
   (justified by Ohm's acoustic law: phase does not affect perceived pitch).
3. **Pitch recognition (original algorithm)** — for each window the **fundamental frequency (F0)** is
   estimated by: detecting spectral peaks (`scipy.signal.find_peaks`); classifying them as
   *relevant / non-relevant* with **dynamic K-Means** (K grows until at most `MAX_REL_PEAKS` remain);
   pre-smoothing noise via the power spectral density (FFT of the windowed autocorrelation); generating
   F0 **candidates** (including differences between peaks, to recover a missing fundamental) and scoring
   each by a **harmonic-count metric**; and mapping to the nearest **equal-tempered pitch**
   (`n(f) = 69 + 12·log₂(f/440)`, MIDI).
4. **Temporal correction** — each estimate is refined using its neighbourhood with **ADSR-inspired
   weighting** (reliability = peak amplitude).
5. **Note & duration extraction** — equal consecutive notes are grouped; durations are computed from the
   window step and **quantized** to musical figures for a given tempo (`QUARTER_PPM`); a silence
   threshold separates note attacks.
6. **Score generation** — notes are assembled with **`music21`** and exported as **MIDI** and **PDF**
   (MusicXML rendered by **MuseScore 4**), with automatic clef selection.

> The system is **monophonic**. The thesis documents **9 successive algorithm iterations**; the final
> "dynamic K-Means + corrections" version is the one shipped here.

**Files:** `w2mc.py` (Tkinter app) · `precision.py` (accuracy evaluation) · `w2mc_times.py` (timing) ·
`Conversor.ipynb`, `ReconocimientoTonos.ipynb` (development notebooks) · `DoM-piano.wav` (test audio).

---

## 2. 2MANYFILTERS — `2MANYFILTERS/`

A standalone desktop app that applies **frequency-domain filters** to a WAV file and writes a new WAV.

- **Filters:** low-pass (LPF), high-pass (HPF), band-pass (BPF), band-stop (BSF); parameters are the
  cutoff thresholds in Hz.
- **Method:** the signal is split into overlapping **Hanning** windows; each window goes through
  **FFT → spectral bins outside the pass band are zeroed → IFFT**, and the audio is rebuilt by
  **overlap-add** (overlap fixed to half the window, a Hanning partition of unity).
- **GUI:** Tkinter.

**Files:** `2mf.py` (Tkinter app) · `Filtros.ipynb` (development notebook) · `DoM-piano.wav` (test audio).

---

## Results / honest framing

- **Main test audio:** `DoM-piano.wav` — an ascending/descending **C-major scale**, piano, ~14 s,
  44.1 kHz (window 0.2 s, overlap 0.15 s).
- **Accuracy of the final algorithm: 99.64% per-window note accuracy** (vs. ~53.6% for the naïve first
  attempt — see the 9-iteration progression in the thesis). **Time complexity: linear, O(n)** in the
  number of windows.
- Other qualitative examples transcribed: *Ode to Joy* (piano), *Spanish anthem* (guitar), *Imperial
  March* (piano).

> ⚠️ **Honest caveat:** the 99.64% is **per-window note accuracy on a single, clean monophonic test
> audio** — *not* a standard MIR benchmark or an F-measure over a public dataset. The system is
> monophonic and can still confuse an octave / fundamental vs. harmonic.

---

## Repository structure

```text
DGIIM-TFG-23-24/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── TFG-JavierGranadosLopez-23-24.pdf   # full thesis (Spanish)
├── WAV2MIDICONVERTER/
│   ├── w2mc.py                   # WAV → score (Tkinter GUI)
│   ├── precision.py              # per-window accuracy evaluation
│   ├── w2mc_times.py             # timing experiments
│   ├── Conversor.ipynb           # development notebook
│   ├── ReconocimientoTonos.ipynb # pitch-recognition notebook
│   ├── clef.png, icon.ico        # GUI assets
│   └── DoM-piano.wav             # test audio
└── 2MANYFILTERS/
    ├── 2mf.py                    # audio filter (Tkinter GUI)
    ├── Filtros.ipynb             # development notebook
    ├── filter.png, icon.ico      # GUI assets
    └── DoM-piano.wav             # test audio
```

---

## Installation & usage

> Both tools are **Tkinter desktop apps**. PDF export in the converter requires **MuseScore 4**
> installed and configured as music21's MusicXML backend. `tkinter` ships with the standard Python
> distribution. Prefer **Python 3.10–3.12**.

```bash
git clone https://github.com/Lhesarel/DGIIM-TFG-23-24
cd DGIIM-TFG-23-24
pip install -r requirements.txt
```

**Run the converter** (a window opens; pick a `.wav`, set window size / overlap / tempo / quantization /
time signature / thresholds, run → MIDI + PDF):
```bash
cd WAV2MIDICONVERTER && python w2mc.py
```

**Run the audio filter** (pick LPF / HPF / BPF / BSF and the cutoff threshold(s) in Hz → filtered `.wav`):
```bash
cd 2MANYFILTERS && python 2mf.py
```

*Prefer not to install anything? Use the [live demos](#-live-demos-no-install) above.*

---

## Building standalone executables (optional)

Both tools can be packaged as single-file executables with [PyInstaller](https://pyinstaller.org/)
(`pip install pyinstaller`). Run the command **inside the corresponding project folder**. The examples
below use the Linux/macOS data separator `:`; **on Windows replace it with `;`**.

**WAV2MIDICONVERTER** (`WAV2MIDICONVERTER/`):
```bash
pyinstaller --onefile --windowed --add-data "clef.png:." --add-data "icon.ico:." --hidden-import "PIL._tkinter_finder" w2mc.py
```
On Linux you may also need to bundle the music21 data: append `--add-binary "<path-to-music21>:."`.

**2MANYFILTERS** (`2MANYFILTERS/`):
```bash
pyinstaller --onefile --windowed --add-data "filter.png:." --add-data "icon.ico:." --hidden-import "PIL._tkinter_finder" 2mf.py
```

> Freshly built PyInstaller binaries are sometimes flagged as false positives by antivirus engines
> (e.g. Windows Defender). The compiled `.exe` are intentionally **not** committed to this repository
> (they are large and AV-flagged) — build them locally, or publish them as a GitHub Release if needed.

---

## Author

**Javier Granados López** — AI/ML Engineer & Researcher · music × machine learning

- Email: javiergrana2lopez@gmail.com
- GitHub: [github.com/Lhesarel](https://github.com/Lhesarel)

Background: Double Degree in Computer Engineering & Mathematics + MSc in Data Science (University of
Granada); *Enseñanzas artísticas superiores de Composición* (Real Conservatorio Superior de Música
Victoria Eugenia de Granada) and professional viola training. Currently a PhD researcher in fairness
in machine learning (University of Granada).

---

## License

Released under the [MIT License](LICENSE).

# VanniSutra 

> **Speak. Measure. Improve.**  
> *Hackathon Track: Contrastive Speech Analytics & Temporal Flaw Grounding*

---

## 🎯 Overview

**VanniSutra** is an AI-powered speech evaluation and communication improvement platform. It helps language learners, job candidates, students, and professionals measure and elevate their speaking skills.

Unlike generic voice coaches that rely on arbitrary, non-deterministic LLM scoring, VocalLens AI uses a **reproducible, contrastive dual-baseline rubric**. It measures your speech against empirical distributions of **ideal exemplars** and **flawed speech**, producing deterministic scores across 8 core speech dimensions and **pinpointing exact timestamped flaws** directly on an interactive audio waveform.

---

## 🏗️ Architecture

```
                    ┌───────────────────────────────────────────────┐
                    │           VanniSutra AI Frontend              │
                    │   (React + Vite + Tailwind + WaveSurfer.js)   │
                    └──────────────────────┬────────────────────────┘
                                           │
                                  Audio Upload / Stream
                                           │
                                           ▼
                    ┌───────────────────────────────────────────────┐
                    │               FastAPI Backend                 │
                    └───────┬──────────────┬──────────────┬─────────┘
                            │              │              │
                16kHz Mono  │              │              │ Word Timestamps
                            ▼              ▼              ▼
                 ┌───────────────┐  ┌─────────────┐  ┌──────────────┐
                 │  PyAV / SF    │  │ Parselmouth │  │faster-whisper│
                 │ Preprocessing │  │  + Librosa  │  │(small, int8) │
                 └───────┬───────┘  └──────┬──────┘  └──────┬───────┘
                         │                 │                │
                         │          Pitch & Energy          │ Word Tokens & Conf
                         └─────────────────┼────────────────┘
                                           ▼
                 ┌──────────────────────────────────────────────────┐
                 │        8-Dimension Speech Metrics Engine         │
                 │ Pace • Pauses • Fillers • Fluency • Clarity •   │
                 │ Pronunciation • Confidence • Vocabulary          │
                 └─────────────────────────┬────────────────────────┘
                                           ▼
                 ┌──────────────────────────────────────────────────┐
                 │          Contrastive Scorer (baseline.json)      │
                 │  score = clip((x - flawed) / (ideal - flawed))   │
                 └─────────────────────────┬────────────────────────┘
                                           ▼
                 ┌──────────────────────────────────────────────────┐
                 │      Temporal Flaw Grounding & Feedback Engine   │
                 │  (long_pause, filler, repetition, speed, mumble) │
                 └─────────────────────────┬────────────────────────┘
                                           ▼
                 ┌──────────────────────────────────────────────────┐
                 │        SQLite Database (SQLAlchemy Models)       │
                 │    users, sessions, dimension_scores, flaw_events │
                 └──────────────────────────────────────────────────┘
```

---

## 📊 Core Speech Dimensions & Rubric

| Dimension | Weight | Measurement Method | Ideal Direction |
|---|---|---|---|
| **Clarity** | 15% | Mean ASR word acoustic confidence ($\bar{p} \in [0, 1]$) | Higher |
| **Fluency** | 15% | Penalty index on repetitions, restarts, and speech gaps | Higher |
| **Pace** | 15% | Overall WPM & 10s sliding window adherence to target (130–155 WPM) | Target-Centric |
| **Pauses** | 15% | Dead silence gaps > 1.2s (high severity if > 2.0s) & pause duration ratio | Lower |
| **Fillers** | 15% | Frequency & rate of verbal crutches (*um, uh, like, basically, you know, matlab, toh*) | Lower |
| **Pronunciation** | 10% | Proportion of high-confidence words (probability $\ge 0.5$) | Higher |
| **Confidence** | 10% | Pitch variation (F0 std dev via Praat), RMS energy stability, hedge words penalty | Higher |
| **Vocabulary** | 5% | Lexical diversity (Type-Token Ratio / TTR) and average word length | Higher |

### Contrastive Scoring Formula
For each metric $x$, the calibrated score is computed strictly against baseline distributions:
$$\text{Score} = \text{clip}\left(\frac{x - \mu_{\text{flawed}}}{\mu_{\text{ideal}} - \mu_{\text{flawed}}}, 0, 1\right) \times 100$$

- **100% Deterministic:** Identical speech always receives the exact same score.
- **Score Bands:**
  - **85.0 – 100.0:** Excellent
  - **70.0 – 84.9:** Good
  - **50.0 – 69.9:** Needs work
  - **0.0 – 49.9:** Weak

---

## 📈 Empirical Validation Benchmark

Calibrated on the paired contrastive sample dataset (`dataset/baseline.json`):

| Sample Clip | Category | Score | Band | Pace (WPM) | Fillers/min |
|---|---|---|---|---|---|
| `ideal/ideal_01_interview.wav` | Ideal | **79.7** | Good | 142.1 | 0.00 |
| `flawed/flawed_01_interview.wav` | Flawed | **22.1** | Weak | 98.4 | 15.82 |
| `ideal/ideal_02_pitch.wav` | Ideal | **79.2** | Good | 145.0 | 0.00 |
| `flawed/flawed_02_pitch.wav` | Flawed | **17.5** | Weak | 94.2 | 18.52 |
| `ideal/ideal_03_leadership.wav` | Ideal | **95.6** | Excellent | 148.5 | 0.00 |
| `flawed/flawed_03_leadership.wav` | Flawed | **10.6** | Weak | 88.0 | 17.12 |
| `ideal/ideal_04_support.wav` | Ideal | **90.5** | Excellent | 140.2 | 2.23 |
| `flawed/flawed_04_support.wav` | Flawed | **23.1** | Weak | 102.1 | 12.98 |
| `ideal/ideal_05_research.wav` | Ideal | **84.4** | Good | 144.8 | 0.00 |
| `flawed/flawed_05_research.wav` | Flawed | **15.9** | Weak | 92.5 | 17.28 |

- **Average Ideal Score:** **85.9 / 100**
- **Average Flawed Score:** **17.8 / 100**
- **Empirical Separation:** **+68.0 points** (Clear, robust statistical differentiation)

---

## 🚀 Quickstart & Setup (Windows & Cross-Platform)

### Prerequisites
- Python 3.10+ (Tested on Python 3.11, 3.12, 3.14)
- Node.js 18+ and npm

### 1. Install Backend Dependencies
```powershell
python -m pip install -r backend/requirements.txt
```

### 2. Configure Environment
```powershell
cp .env.example .env
```

### 3. Generate Sample Dataset & Calibrate Baseline
```powershell
python -m backend.scripts.generate_sample_dataset
python -m backend.scripts.build_baseline
python -m backend.scripts.seed_demo
```

### 4. Install Frontend Dependencies
```powershell
cd frontend
npm install
cd ..
```

### 5. Launch Both Servers in One Command
- **Windows:** Double-click or run:
  ```powershell
  .\run.bat
  ```
- **Linux / macOS:**
  ```bash
  chmod +x run.sh
  ./run.sh
  ```

Access the app in your browser at: **`http://localhost:5173`**  
Interactive Swagger API docs at: **`http://127.0.0.1:8000/docs`**

---

## 🧪 Running the Test Suite

```powershell
python -m pytest backend/tests/ -v
```

All 9 comprehensive tests pass:
1. `test_health_check_endpoint`: System health and model availability
2. `test_baseline_endpoint`: Baseline distributions verification
3. `test_sessions_list_endpoint`: Session query integrity
4. `test_session_detail_schema`: Full ReportResponse schema compliance
5. `test_progress_endpoint_schema`: Longitudinal tracking contract
6. `test_analyze_endpoint_upload`: Multipart audio file analysis workflow
7. `test_scoring_determinism`: Proves identical input yields identical output
8. `test_detector_finds_known_flaws`: Ground-truth flaw grounding verification
9. `test_ideal_scores_higher_than_flawed`: Contrastive score separation guarantee

---

## 🎬 Judge Demo Walkthrough (No Mic Required)

1. Launch the application (`.\run.bat`).
2. Open `http://localhost:5173`.
3. In the top navbar, click **Demo Mode**:
   - Select **Flagship Presentation**: Observe the **89.5 Excellent** score, balanced radar chart, high vocabulary, and clean waveform with 0 critical flaws.
   - Select **Disfluent Speech Sample**: Observe the **51.8 Needs work** score, and see red/amber **temporal flaw markers** along the audio timeline.
   - Click on any red marker (e.g. at 3.2s) or any highlighted red word in the transcript: Notice how the **WaveSurfer audio player instantly jumps** directly to that second!
4. Navigate to **Contrastive Compare**: Review the side-by-side bar chart showing User vs. Ideal Exemplar vs. Flawed Exemplar.
5. Navigate to **Progress Analytics**: Inspect the longitudinal improvement trendline over the 5 practice sessions.
6. Navigate to **Dataset & Rubric**: Listen directly to the paired synthetic audio exemplars and inspect the mathematical baseline distributions.

---

## 🔑 Groq API Key Configuration (Audio-to-Text Transcription)

VocalLens AI integrates the **Groq Whisper Audio API** (`whisper-large-v3-turbo` / `whisper-large-v3`) for ultra-low latency, high-accuracy speech-to-text transcription on the server, paired with **Gemini AI** for deep contrastive speech analysis, acoustic scoring, and personalized coaching.

### Where to add your `GROQ_API_KEY`:

1. In the project root directory, open or create your `.env` file:
   ```env
   GROQ_API_KEY=gsk_your_actual_groq_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
2. In **Google AI Studio** / Cloud Run environment:
   - Add `GROQ_API_KEY` to the **Environment Secrets / Variables** configuration.
   - The key is securely accessed exclusively on the backend (`POST /api/transcribe` and `POST /api/analyze`) and is never exposed to the client or browser JavaScript.
3. Restart or reload your server process to load the environment variable.

# Clear Voice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repo that measures, honestly, how well fine-tuned whisper-small transcribes dysarthric speakers and sentences it never heard. It also produces the results, figures and demo app for the Veritas submission.

**Architecture:**
- A small Python package `clearvoice/` with pure functions, tested on CPU with tiny synthetic data.
- Five thin Colab notebooks that call the package on the real TORGO data on a GPU.
- A Gradio app for a Hugging Face Space.
- Everything that needs no GPU (data checks, splits, tests, smoke runs with whisper-tiny) runs on the local Mac.

**Tech Stack:**
- Python 3.12, PyTorch, Hugging Face `transformers` (Whisper and Seq2SeqTrainer), `huggingface_hub` and `pyarrow`.
- No `datasets` library: its v4 audio decoding needs ffmpeg and torchcodec.
- `soundfile` and `scipy` for audio, `jiwer`, `pandas`, `matplotlib`, `gradio`, `pytest`, `nbformat`.

**Spec:** `docs/superpowers/specs/2026-11-14-clear-voice-design.md`

**Execution:** native, in this session. The user approved the design with "start building" and is away until tonight.

## Global Constraints

- Dataset: `abnerh/TORGO-database`, pinned to revision `6e9b33c1af30eb3878574a1c1f5e60a60bd6d9c5`, files `data/train-0000{0..3}-of-00004.parquet`.
- Never commit audio, parquet files or model weights. Commit only code, row-index split CSVs, result CSVs and figures.
- Speaker folds, verbatim from the spec:
  - Fold A: dysarthric F01, M01, F03, M03; controls FC01, MC01, MC02.
  - Fold B: dysarthric M02, M04, M05, F04; controls FC02, FC03, MC03, MC04.
- Prompt split: 70% train, 10% validation, 20% test, by unique normalised prompt, stratified by single word versus sentence. Seed 42 everywhere.
- Evaluation uses one copy per utterance (prefer `headMic`, else `arrayMic`). Training uses all copies.
- Transcription settings: `language="en"`, `task="transcribe"`, no timestamps. Normalise references and outputs with Whisper's `EnglishTextNormalizer` (`WhisperTokenizer.normalize`).
- Fine-tune defaults: `openai/whisper-small`, learning rate 1e-5, batch 16, warmup 10%, `max_steps` 2000, eval and save every 250 steps on validation WER, keep the best, `save_total_limit=1`, save model weights only, bf16 when CUDA supports it.
- Personalisation: 2 speakers (one severe, one mild or moderate), k ∈ {2, 5, 10} minutes, measured on one copy per utterance; a k above a speaker's total is recorded as `capped` and skipped.
- Every Health or medical-sounding screen and the README say: "Research prototype, not a medical device. Uses the TORGO database (non-commercial research use)."
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

Likely failure modes that no other task's tests exercise:

1. **Microphone input from the demo** arrives as `(sample_rate, int16 array)`, often 44.1 or 48 kHz and stereo. Expected: converted to float32 mono at 16 kHz, never passed through raw.
2. **Dataset audio not 16 kHz mono.** Expected: resampled and mixed to mono when decoded.
3. **A filename that doesn't match** `{speaker}_{session}_{mic}_{utt}.wav`. Expected: a loud error naming the bad paths, never silently dropped rows.
4. **Transcripts that normalise to empty, or a model that outputs nothing.** Expected: empty references are dropped from scoring and counted; empty or runaway outputs are scored without crashing.
5. **A personalisation request for more minutes than a speaker has.** Expected: marked `capped`, never silently trained on less.

---

### Task 1: Project setup, text normalisation and filename parsing

**Files:**
- Create: `.gitignore`, `requirements.txt`, `requirements-dev.txt`, `pyproject.toml` (package `clearvoice`, pytest config)
- Create: `clearvoice/__init__.py`, `clearvoice/text.py`, `clearvoice/data.py` (parsing part)
- Test: `tests/test_text.py`, `tests/test_parse.py`

**Interfaces:**
- Produces:
  - `text.normalize(s: str) -> str`: Whisper English normaliser, loaded once, lazily.
  - `text.prompt_type(norm: str) -> Literal["word", "sentence"]`
  - `data.parse_path(path: str) -> dict` with keys `speaker`, `session`, `mic`, `utt`, `utt_key` (`f"{speaker}_{session}_{utt}"`). Raises `ValueError` on a non-matching basename.
  - Constants `DATASET_ID`, `REVISION`, `PARQUET_FILES`, `DYSARTHRIC`, `CONTROLS`, `SEVERITY`.

- [ ] **Step 1: Set up the environment.** Create a `.venv` with python3.12, `git init`, and pip install dev requirements.
- [ ] **Step 2: Write the failing tests.**
  - `parse_path("FC02_3_headMic_0006.wav")` gives `speaker="FC02"`, `session="3"`, `mic="headMic"`, `utt="0006"`, `utt_key="FC02_3_0006"`.
  - A directory prefix is ignored.
  - `parse_path("bad.wav")` raises `ValueError`.
  - `normalize("Seven, Seven!") == normalize("seven seven")`.
  - `prompt_type("yes") == "word"` and `prompt_type("the quick brown fox") == "sentence"`.
- [ ] **Step 3: Run the tests and confirm they fail.** `pytest tests/test_text.py tests/test_parse.py -q` should fail on import errors.
- [ ] **Step 4: Implement.**
  - The filename regex is `^(?P<speaker>[FM]C?\d{2})_(?P<session>[^_]+)_(?P<mic>[A-Za-z]+Mic)_(?P<utt>\d+)\.wav$`.
  - `normalize` uses `WhisperTokenizer.from_pretrained("openai/whisper-tiny").normalize`, cached.
- [ ] **Step 5: Run the tests and confirm they pass,** then commit with `feat: project setup, text normalisation, filename parsing`.

### Task 2: Metadata loading and audio access

**Files:**
- Modify: `clearvoice/data.py`
- Create: `clearvoice/audio.py`
- Test: `tests/test_data.py`, `tests/test_audio.py`, `tests/conftest.py` (fixture writing a 2-file synthetic parquet set with WAV bytes in the same schema as TORGO)

**Interfaces:**
- Consumes: `parse_path`, `normalize`, `prompt_type`.
- Produces:
  - `data.download_parquets(cache_dir: str | None = None) -> list[Path]`
  - `data.load_metadata(parquet_paths: list[Path]) -> pd.DataFrame`. Columns: `row_id, file_idx, row_in_file, path, speaker, session, mic, utt, utt_key, text, text_norm, prompt_type, n_words, speech_status, is_dysarthric, gender, duration, severity`. `row_id` is the position in concatenated file order.
  - `data.load_audio(parquet_paths, meta: pd.DataFrame, row_ids: Iterable[int]) -> dict[int, np.ndarray]`: float32, mono, 16 kHz.
  - `audio.decode_wav(b: bytes) -> np.ndarray`: 16 kHz mono float32.
  - `audio.to_16k_mono(x: np.ndarray, sr: int) -> np.ndarray`: accepts int16 or float, 1-D or 2-D.
  - `data.check_mic_pairs(meta) -> pd.DataFrame`: the utterance groups whose copies disagree on `text_norm`.

- [ ] **Step 1: Write the failing tests.**
  - `load_metadata` on the fixture returns every row, with `row_id` equal to `range(n)` and the parsed fields filled.
  - `load_audio` returns the requested ids only, as 1-D float32 at 16 kHz.
  - A fixture row with a bad filename makes `load_metadata` raise a `ValueError` that names the file. **(Review Focus 3)**
  - `to_16k_mono` on a stereo int16 48 kHz array of 1 s returns about 16,000 float32 samples within [-1, 1]. **(Review Focus 1, 2)**
  - `decode_wav` on 8 kHz WAV bytes returns 16 kHz.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement.**
  - Iterate with `pyarrow.parquet.ParquetFile.iter_batches(batch_size=256)`, dropping audio bytes for metadata.
  - Resample with `scipy.signal.resample_poly`.
  - `download_parquets` uses `hf_hub_download(repo_type="dataset", revision=REVISION)`.
- [ ] **Step 4: Run the tests and confirm they pass.**
- [ ] **Step 5: Run on the real data locally.** Download, `load_metadata`, then print:
  - speakers and clip counts,
  - minutes per speaker,
  - microphone counts,
  - the sample rates seen in 50 decoded clips,
  - the number of empty `text_norm` rows,
  - the `check_mic_pairs` count.

  Save the report to `results/data_report.md`.
- [ ] **Step 6: Commit** with `feat: metadata loading, audio decoding, data report`.

### Task 3: Splits

**Files:**
- Create: `clearvoice/splits.py`, `splits/*.csv` (generated), `scripts/make_splits.py`
- Test: `tests/test_splits.py`

**Interfaces:**
- Consumes: the `load_metadata` DataFrame.
- Produces:
  - `assign_prompt_splits(meta, seed=42) -> dict[str, str]`: `text_norm` mapped to `"train" | "val" | "test"`.
  - `pick_eval_copies(df) -> pd.DataFrame`: one row per `utt_key`, preferring `headMic`.
  - `FOLDS: dict[str, dict[str, list[str]]]`
  - `fold_splits(meta, prompt_split, heldout: str) -> dict[str, pd.DataFrame]`. Keys:
    - `train`: other fold's speakers × train prompts, all copies.
    - `val`: other fold × val prompts, eval copies.
    - `test_seen`: held-out fold × train prompts, eval copies.
    - `test_strict`: held-out fold × test prompts, eval copies.
    - `test_newsent`: other fold × test prompts, eval copies.
  - `random_splits(meta, seed=42) -> dict[str, pd.DataFrame]`: 80/10/10 by row (stretch).
  - `personal_subsets(meta, prompt_split, speaker, minutes=(2, 5, 10), seed=42) -> dict[int, pd.DataFrame | Literal["capped"]]`. Nested prefixes of a shuffled pool of eval copies with train prompts.
  - `write_splits(splits: dict, out_dir: Path, name: str)`. CSV columns: `row_id, path, speaker, utt_key, mic, prompt_type, duration`, with no transcripts.

- [ ] **Step 1: Write the failing tests** on a synthetic metadata frame with 4 speakers, 2 mics and 30 prompts:
  - Prompt splits are disjoint and cover every prompt, at about 70/10/20 per prompt type.
  - No `val` or `test` prompt appears in any `train` frame.
  - The held-out speakers never appear in `train` or `val`.
  - `pick_eval_copies` leaves `utt_key` unique and keeps `headMic` when present.
  - Results are deterministic for a fixed seed.
  - `personal_subsets` returns nested sets whose durations reach each k, and returns `"capped"` when k exceeds the pool. **(Review Focus 5)**
  - The personalisation pool never contains a test prompt.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and confirm they pass.**
- [ ] **Step 5: Run `scripts/make_splits.py` on the real metadata.** It writes `splits/` and appends to `results/data_report.md`, per fold: clips per part, speakers per part, strict-test clips per speaker, and personalisation minutes available.
- [ ] **Step 6: Commit** with `feat: leakage-safe splits`.

### Task 4: Metrics

**Files:**
- Create: `clearvoice/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `score(refs: list[str], hyps: list[str]) -> dict`: keys `wer`, `cer`, `n`, `n_dropped_empty_ref`. Corpus-level, after `normalize`.
  - `table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame`: `df` has `ref`, `hyp`, `speaker` and the group columns. Output columns: `wer`, `cer`, `n_clips`, `n_speakers`.
  - `speaker_bootstrap(df, metric: str = "wer", n_boot: int = 1000, seed: int = 42) -> tuple[float, float]`: a 95% interval from resampling speakers.

- [ ] **Step 1: Write the failing tests.**
  - Identical texts score `wer == 0`.
  - `score(["a b c d"], ["a x c"])` gives `wer == 0.5`.
  - An empty reference is dropped and counted. **(Review Focus 4)**
  - An empty output scores `wer == 1.0` without error.
  - A runaway output scores `wer > 1` without error. **(Review Focus 4)**
  - `table` counts clips and speakers per group.
  - The bootstrap interval brackets the point estimate on toy data.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement with `jiwer`.**
- [ ] **Step 4: Run the tests and confirm they pass.**
- [ ] **Step 5: Commit** with `feat: WER/CER metrics and speaker bootstrap`.

### Task 5: Transcription

**Files:**
- Create: `clearvoice/asr.py`
- Test: `tests/test_asr.py` (marked `slow`: downloads whisper-tiny)

**Interfaces:**
- Consumes: `audio.to_16k_mono`.
- Produces:
  - `load_model(name_or_path: str, device: str | None = None) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]`. Device is CUDA when available, else CPU (never MPS). Uses fp16 on CUDA.
  - `transcribe(model, processor, audios: list[np.ndarray], batch_size: int = 16) -> list[str]`
  - `transcribe_rows(model, processor, rows: pd.DataFrame, audio: dict[int, np.ndarray], batch_size=16) -> pd.DataFrame`: `rows` plus `hyp` and `ref`.

- [ ] **Step 1: Write the failing test.** whisper-tiny on two synthetic clips (1 s of silence, 2 s of tone) returns 2 strings. `transcribe_rows` keeps the row order.
- [ ] **Step 2: Run the test and confirm it fails.**
- [ ] **Step 3: Implement** with `generate(input_features, language="en", task="transcribe", max_new_tokens=128)`.
- [ ] **Step 4: Run the test and confirm it passes.** `pytest -m slow tests/test_asr.py`
- [ ] **Step 5: Commit** with `feat: whisper transcription helpers`.

### Task 6: Fine-tuning

**Files:**
- Create: `clearvoice/train.py`
- Test: `tests/test_train.py` (marked `slow`)

**Interfaces:**
- Consumes: `asr.load_model`, `metrics.score`.
- Produces:
  - `@dataclass TrainConfig`. Defaults from Global Constraints, plus `num_train_epochs: float | None = None` (used when `max_steps` is `None`) and `eval: bool = True`.
  - `finetune(cfg: TrainConfig, train_rows: pd.DataFrame, val_rows: pd.DataFrame | None, audio: dict[int, np.ndarray], output_dir: str) -> str`: returns a directory holding the best model and its processor.

- [ ] **Step 1: Write the failing test.** whisper-tiny, 4 synthetic clips, `max_steps=2`, `eval_steps=1`, CPU. The returned directory loads with `load_model` and `transcribe` returns strings.
- [ ] **Step 2: Run the test and confirm it fails.**
- [ ] **Step 3: Implement.**
  - The collator computes log-mel features from the raw arrays in each batch and pads labels with -100, as in Hugging Face's Whisper fine-tuning blog.
  - `use_cpu=not torch.cuda.is_available()`.
  - `save_only_model=True`.
  - Validation WER is computed through `metrics.score`.
  - Use `EarlyStoppingCallback(patience=3)` when `eval`.
- [ ] **Step 4: Run the test and confirm it passes.**
- [ ] **Step 5: Commit** with `feat: whisper fine-tuning`.

### Task 7: Results tables and figures

**Files:**
- Create: `clearvoice/report.py`, `clearvoice/plots.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `metrics.table`, `metrics.speaker_bootstrap`.
- Produces:
  - `headline(preds: dict[str, pd.DataFrame]) -> pd.DataFrame`, keyed by system name. One row per system: strict sentence WER, strict word CER, 95% interval, clip and speaker counts.
  - `leakage(preds_ft: pd.DataFrame) -> pd.DataFrame`, over `test_seen`, `test_newsent` and `test_strict`.
  - `per_speaker(preds: dict) -> pd.DataFrame`
  - `worst_clips(preds_ft, n=10) -> pd.DataFrame`, by CER on strict.
  - `pick_demo_clips(preds_ft, per_severity=5) -> pd.DataFrame`, strict only.
  - `plots.bar_systems(df, path)`, `plots.leakage(df, path)`, `plots.personalisation(df, path)`, `plots.per_speaker(df, path)`. Each writes a PNG.

- [ ] **Step 1: Write the failing tests** on fake prediction frames:
  - `headline` has one row per system.
  - `worst_clips` returns only strict rows, sorted.
  - `pick_demo_clips` returns only strict rows.
  - Every plot function writes a non-empty file.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement.** Follow the dataviz skill for colours and labels.
- [ ] **Step 4: Run the tests and confirm they pass.**
- [ ] **Step 5: Commit** with `feat: results tables and figures`.

### Task 8: Colab notebooks

**Files:**
- Create: `scripts/build_notebooks.py` (nbformat), which generates `notebooks/01_data_and_splits.ipynb` to `05_results_and_failures.ipynb`
- Test: `tests/test_notebooks.py`

**Interfaces:**
- Consumes: every package function above.
- Produces: notebooks. Each starts with a setup cell that:
  - clones `REPO_URL`,
  - runs `pip install -r requirements.txt`,
  - mounts Drive at `/content/drive`,
  - sets `WORK = /content/drive/MyDrive/clearvoice` for models and results,
  - reads the Hugging Face token from Colab Secrets (`userdata.get("HF_TOKEN")`).

  What each notebook does:
  - **01** reproduces Tasks 2 and 3.
  - **02** runs zero-shot whisper-small and whisper-large-v3 on `test_seen`, `test_strict` and `test_newsent` for both folds, plus 200 LibriSpeech `clean/test/0000.parquet` clips, and writes `preds_zeroshot_*.csv`.
  - **03** has a `HELDOUT = "A"` parameter. It runs `finetune`, then predicts on the fold's test parts and LibriSpeech, writes `preds_ft_hold{HELDOUT}.csv`, and optionally pushes the model to the Hub.
  - **04** personalises 2 speakers and writes `personalise.csv`.
  - **05** builds the tables, figures, `worst_clips.csv` and `app/demo_clips.csv`.

- [ ] **Step 1: Write the failing test.** Each notebook parses with nbformat. Every `clearvoice.` name referenced in code cells exists in the package (checked by AST plus `hasattr`).
- [ ] **Step 2: Run the test and confirm it fails.**
- [ ] **Step 3: Implement `build_notebooks.py`** and generate the notebooks.
- [ ] **Step 4: Run the test and confirm it passes.**
- [ ] **Step 5: Dry-run notebook 02's and 03's core cells locally on 6 strict clips with whisper-tiny,** using a `SMOKE=True` flag in the notebooks.
- [ ] **Step 6: Commit** with `feat: colab notebooks`.

### Task 9: Demo app

**Files:**
- Create: `app/app.py`, `app/requirements.txt`, `app/README.md` (Space front matter)
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `asr.load_model`, `asr.transcribe`, `audio.to_16k_mono`, `data.download_parquets`, `data.load_audio`, `app/demo_clips.csv`.
- Produces:
  - `prepare_mic(x: tuple[int, np.ndarray] | None) -> np.ndarray | None`
  - `build_ui() -> gr.Blocks`, with 3 tabs.
  - Model ids come from env vars `CV_MODEL_A` and `CV_MODEL_B`, falling back to `openai/whisper-small` with a visible "fine-tuned model not loaded" note.

- [ ] **Step 1: Write the failing tests.**
  - `prepare_mic((48000, stereo_int16))` returns 16 kHz mono float32. **(Review Focus 1)**
  - `prepare_mic(None)` returns `None`.
  - `build_ui()` builds without loading models, which load lazily.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement.** The disclaimer from Global Constraints goes on every tab.
- [ ] **Step 4: Run the tests and confirm they pass.** Then launch locally once with the whisper-tiny fallback and check one microphone file and one demo clip.
- [ ] **Step 5: Commit** with `feat: gradio demo app`.

### Task 10: Write-up, outreach and handover

**Files:**
- Create: `README.md` (Veritas template plus setup), `docs/therapist_email.md`, `docs/TONIGHT.md` (the user's checklist)

- [ ] **Step 1: Write `README.md`** in the template sections from the spec's appendix.
  - Results are placeholders marked `TODO(results)`.
  - Include the track-fit line, the "What we added" text, the citations and credits, and the disclaimer.
- [ ] **Step 2: Write `docs/therapist_email.md`,** a short, plain email asking which transcription errors matter most to their clients.
- [ ] **Step 3: Write `docs/TONIGHT.md`.**
  - The user's steps: Colab Pro, the HF account and token in Colab Secrets, Drive access, the GitHub repo.
  - The run order: notebooks 02, then 03 with A, then 03 with B.
  - The expected durations.
  - How to get results back into the repo.
- [ ] **Step 4: Run the full fast test suite.** `pytest -q -m "not slow"` passes; then `pytest -q -m slow` passes.
- [ ] **Step 5: Commit** with `docs: write-up draft, outreach email, tonight checklist`.

### Task 11 (if time): Local zero-shot whisper-small baseline on CPU

- [ ] **Step 1:** Run notebook 02's whisper-small cells as a script on this Mac's CPU, on both folds' `test_strict`, in the background. Write `results/preds_zeroshot_whisper-small.csv`.
- [ ] **Step 2: Commit** with `results: zero-shot whisper-small baseline`.

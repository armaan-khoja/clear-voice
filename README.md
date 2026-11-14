# Clear Voice

**Track:** Languages
**Team:** TODO(team names)

> Research prototype, not a medical device. Uses the TORGO database (non-commercial research use).

## The problem

Speech technology serves some speakers poorly. People whose speech is affected by cerebral palsy, ALS, stroke or Parkinson's (dysarthria) are among the worst served: phone dictation and live captions turn what they say into nonsense, so a tool everyone else uses to text a friend or follow a conversation shuts them out.

## What we built

A speech recogniser fine-tuned for dysarthric speech, and an honest test of it.
- **Input:** a recording of someone speaking.
- **Output:** the words they said.
- **Who it's for:** a 17-year-old with cerebral palsy dictating a text message, and the speech therapist who wants to know whether a few minutes of her recordings would make the tool work for her.
- **Demo:** a web app with three tabs. It plays clips from speakers the model never heard, shows stock Whisper's output beside Clear Voice's and the true words, and lets anyone record themselves.

## What we added

We started from the suggested project **4.4.3, "Speech recognition across accents"** (Whisper, word error rate per group) and its suggested fix of fine-tuning on a small set of clips. We added:

- **Our own user:** people with dysarthria, a group speech recognition serves even worse than accents.
- **Our own data:** the TORGO database of dysarthric and typical speakers, not Common Voice or FLEURS.
- **Our own questions:**
  1. Does it work on a **new speaker saying a new sentence**? TORGO repeats the same ~960 prompts across speakers, so a random split lets a model memorise sentences. We report a strict split where neither the speaker nor the sentence was seen in training, and show how much easier splits inflate the score. Earlier research found the same problem (NP-TORGO); we build on it.
  2. **How many minutes of a person's own voice** does it take to personalise the model?
  3. Did fine-tuning **break everyday speech**?

## How it works

**Data.** We used TORGO (Hugging Face mirror `abnerh/TORGO-database`):
- 16,552 clips from 8 dysarthric speakers (cerebral palsy or ALS; severity from mild to severe) and 7 control speakers.
- Single words (digits, commands, assessment word lists) and sentences (959 unique prompts).
- Every utterance was recorded on two microphones. We train on both copies but score one copy per utterance, so nothing is counted twice.
- We dropped 2 clips longer than Whisper's 30-second window.

**Splits:**
- **Prompts:** split 70% train, 10% validation, 20% test, separately for words and sentences. Test and validation sentences never appear in training for anyone.
- **Speakers:** split into two folds, each with two severe, one moderate and one mild dysarthric speaker:
  - Fold A: F01, M01, F03, M03.
  - Fold B: M02, M04, M05, F04.
- **How the folds are used:** we train one model per fold and test it on the other fold's speakers.
- **Choosing the checkpoint:** validation uses only validation sentences, so picking a checkpoint never rewards memorised sentences.

**Model.** We fine-tuned OpenAI's `whisper-small`:
- Learning rate 1e-5, batch 16, up to 2,000 steps on an L4 GPU.
- The checkpoint is chosen by validation word error rate.

**Baselines:**
- Stock `whisper-small` with no fine-tuning (the simplest guess).
- Stock `whisper-large-v3`, a model about 6× bigger.

**Scoring.** Both the model's output and the true text are normalised (case, punctuation, number words) before scoring:
- **Sentences:** word error rate.
- **Single words:** character error rate.
- **Uncertainty:** 95% intervals come from resampling whole speakers.
- **Medians:** we also report median per-clip error, because Whisper sometimes invents long strings of words for a single word, and one such clip can dominate a total.

## Results

TODO(results): fill from `results/summary.md` after the notebooks run.

**Headline.** Dysarthric speakers the model never heard, saying sentences it never heard:

| System | Sentences: word error rate | Single words: character error rate |
|---|---|---|
| whisper-small (baseline) | TODO | TODO |
| whisper-large-v3 | TODO | TODO |
| **Clear Voice** | TODO | TODO |

**Leakage.** The same Clear Voice model on easier tests: new speaker with seen sentences TODO; seen speaker with new sentences TODO; new speaker with new sentences TODO.

**Personalisation.** TODO (M01, severe; F03, moderate; 0, 2, 5 and 10 minutes).

**Everyday English** (LibriSpeech, 200 clips): whisper-small TODO, Clear Voice TODO.

**Where it fails.** TODO: the worst clips, explained (for example severe speakers on single words, and stock Whisper inventing "thank you" for a word it can't hear).

## What we'd do next

1. Test with a speech therapist and with people with dysarthria who consent to it, on real dictation rather than read prompts.
2. Train on a larger, consented dysarthric speech collection (such as the Speech Accessibility Project) so the model sees more than 4 dysarthric speakers per fold.

## Sources

**Datasets:**
- Rudzicz, F., Namasivayam, A. K. and Wolff, T. (2012). The TORGO database of acoustic and articulatory speech from speakers with dysarthria. *Language Resources and Evaluation* 46(4). Mirror: https://huggingface.co/datasets/abnerh/TORGO-database
- Panayotov, V. et al. (2015). LibriSpeech: an ASR corpus based on public domain audio books. https://huggingface.co/datasets/openslr/librispeech_asr

**Models:**
- OpenAI Whisper (Radford et al., 2023, "Robust Speech Recognition via Large-Scale Weak Supervision"): `openai/whisper-small`, `openai/whisper-large-v3`, `openai/whisper-tiny` (tests only).

**Related work:**
- arXiv 2411.00980, the NP-TORGO split with no shared prompts. TODO: confirm title and authors.
- Google Project Euphonia.
- The Speech Accessibility Project (University of Illinois).
- A TU Delft thesis on LoRA fine-tuning of Whisper on TORGO.

**Tutorials:**
- Hugging Face, "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers".

**Libraries:** PyTorch, Hugging Face Transformers, huggingface_hub, jiwer, pandas, PyArrow, SciPy, soundfile, Matplotlib, Gradio, pytest.

**AI tools:**
- Claude (Anthropic) helped brainstorm the idea, write and test the code, and draft this write-up. We reviewed every part.

## Reproduce

1. Open the notebooks in Google Colab with an L4 GPU and run them in order:
   - `notebooks/01_data_and_splits`
   - `02_zero_shot_baselines`
   - `03_finetune` (with `HELDOUT = "A"`, then `"B"`)
   - `04_personalise`
   - `05_results_and_failures`

   Results and models are saved to Google Drive under `MyDrive/clearvoice`.
2. Run the tests locally:

   ```bash
   python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
   ```

   ```bash
   .venv/bin/pytest
   ```

   ```bash
   .venv/bin/pytest -m slow
   ```

   The slow tests download whisper-tiny.

3. `splits/` holds the row indices of every split. `scripts/make_splits.py` rebuilds them, and notebook 1 checks the rebuild matches.

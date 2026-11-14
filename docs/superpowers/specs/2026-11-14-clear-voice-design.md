# Clear Voice: design

Date: 2026-11-14. Status: approved by the user, 2026-11-14 afternoon.

## Intent and why

Speech-to-text works badly for people with dysarthria: speech changed by cerebral palsy, ALS, stroke or Parkinson's. Clear Voice fine-tunes OpenAI's Whisper so these speakers are understood, and tests it honestly on speakers **and** sentences the model has never heard.

It is built to:
1. **Win the Veritas AI International High School Hackathon, Languages track.** Submissions close Sun Nov 15, 11:59 PM ET. The final ranking is 70% live finals (Nov 21) and 30% Round 1. Round 1 is scored on technical execution (30), originality (20), impact and track fit (20), testing and honesty (15) and communication (15).
2. **Be the proof of concept for the Conrad Challenge Innovation Stage** (Health and Nutrition category, due Jan 7, 2027). The Conrad business materials are out of scope here.

**Users.** The primary user is a person with dysarthria dictating a text message, for example a 17-year-old with cerebral palsy whose phone dictation turns her texts into nonsense. The secondary user is the speech-language therapist who wants to know whether a few minutes of a client's recordings can make the tool work for them.

**Who builds it.** The user builds alone, with Claude writing code locally and the user running it on Colab Pro. Teammates join at the end to combine work (write-up, video, Conrad materials).

## Goals

1. **Headline result.** Word error rate (WER) and character error rate (CER) for fine-tuned whisper-small, against zero-shot whisper-small and zero-shot whisper-large-v3. Measured on held-out speakers × held-out prompts (the strict split).
2. **Leakage table.** The fold models scored on new speakers with seen sentences against new speakers with new sentences. This shows how much of a "too good" number is memorised sentences. *Stretch:* a third row from a random-split fine-tune.
3. **Personalisation curve.** Strict-split error for 2 held-out dysarthric speakers (one severe, one mild or moderate) after adding 0, 2, 5 and 10 minutes of their own recordings (training prompts only), capped at what each speaker has. *Stretch:* up to 4 speakers and a 20-minute point.
4. **Breakdowns.** Results per named speaker and per severity, for single words versus sentences, and for control speakers, to show typical speech didn't get worse. Also a 200-clip general-speech check (LibriSpeech test-clean, streamed), because judges will speak everyday sentences into the demo.
5. **Failure analysis.** The 10 worst strict-split clips, listened to and sorted into causes.
6. **Public demo.** A Hugging Face Space that survives until the Nov 21 finals.
7. **Veritas submission.** A public GitHub repo with notebooks saved with outputs; a README that doubles as the one-page write-up; a 2-minute video; credits.

## Non-goals

- No new model architecture.
- No recording of people with dysarthria.
- No phone app, streaming or real-time captioning.
- No clinical claims.
- No commercial use (TORGO is non-commercial).
- No language-model post-correction.
- No use of the full University of Toronto TORGO release, unless the Hugging Face mirror fails.

## Anti-goals (failure even if it technically works)

- **A leaky headline:** a number taken from a random split or from seen sentences.
- **Claiming the leakage problem as our discovery.** NP-TORGO (arXiv 2411.00980) published it first; cite it and position the work as building on it.
- **A rigged demo:** one that only looks good on cherry-picked clips the model was trained on.
- **Unexplainable code:** any code the user cannot explain line by line under live questioning.

## Constraints

- **Time.** The build starts about 3 PM ET Sat Nov 14, leaving about 29 hours to the first submission, sleep included. Scope is planned reduced from the start; the stretch items come back only if work is ahead of schedule. Hard anchors:
  - first submission by **Sun 8 PM ET**,
  - final resubmission by **Sun 10 PM ET** (the FAQ says to submit at least an hour early and allows no exceptions).
- **Compute.**
  - Training and inference run on Colab Pro. The L4 GPU is the default (24 GB, bf16); use an A100 only if the L4 is too slow.
  - Plain Pro has no background execution, so overnight training needs the browser tab left open. Check the compute-units counter before each long run.
  - Save model weights only, without optimizer state, and keep at most one checkpoint per run, so Drive needs about 5 GB free.
  - The local Mac (M2, 8 GB RAM, about 13 GB free disk) is for writing code and git only, not training.
- **Round 1 judges see only what was submitted before the deadline.** Tag the repo `v1-submission` at submission. Work for the Nov 21 finals happens on a branch afterwards and is disclosed as post-deadline; it is out of scope for this design.
- **Hackathon rules.**
  - All project code is written after 12:00 AM ET Nov 14.
  - Every dataset, model and AI tool is credited, including Claude as a coding assistant.
  - The user must understand every part.
  - The Languages track requires results per accent or speaker group, model output shown beside the correct output, and public data only.
  - No recording anyone else's voice without permission.
- **Data licence.** TORGO is free for non-profit academic use; cite Rudzicz, Namasivayam and Wolff (2012). The Hugging Face mirror has no licence tag. Never commit audio or the dataset to GitHub; commit only row indices, transcripts in result files, metrics and figures.
- **Conrad.** Only Conrad team members' work may go into the Conrad submission. Disclose that the prototype came from Veritas.

## Approaches considered

| | Approach | Trade-off |
|---|---|---|
| A | One big Colab notebook | Simplest to submit; a disconnect loses state; serial work only |
| B | Numbered notebooks sharing a Google Drive folder | Robust and parallel; some duplicated helper code |
| **C (chosen)** | **GitHub repo of small Python modules, with thin Colab notebooks that import them** | Most professional and easiest to extend for Conrad; about an hour of setup; more pieces to explain |

The user picked C. Mitigation for its cost: keep modules small, with plain docstrings and comments, so each notebook runs one stage.

## Data facts (verified 2026-11-14)

- **Source.** Hugging Face `abnerh/TORGO-database`: not gated, 4 parquet files, 1.56 GB, 16,552 rows, a single `train` split.
- **Columns.**

  | Column | Contents |
  |---|---|
  | `audio` | Includes the original file path; 16 kHz |
  | `transcription` | Punctuation removed, case kept |
  | `speech_status` | `dysarthria` or `healthy` |
  | `gender` | |
  | `duration` | |
  | `severity` | Speaker-level |

- **Content.** Short words (digits, commands, radio alphabet, Frenchay and Yorkston word lists) and restricted sentences (phoneme-rich sentences, the Grandfather passage, Yorkston sentences, 460 MOCHA-TIMIT prompts). About 957 unique prompts, heavily repeated across speakers.
- **Speaker IDs.** Parsed from the path `{speaker}_{session}_{mic}_{utt}.wav`, for example `FC02_3_headMic_0006.wav`.
  - Dysarthric: F01, F03, F04, M01, M02, M03, M04, M05.
  - Controls: FC01–FC03, MC01–MC04.
- **Microphone pairs.** The same utterance often exists as both `headMic` and `arrayMic`. The group key for one utterance is (speaker, session, utt).
- **Severity.**

  | Severity | Speakers |
  |---|---|
  | Severe | F01, M01, M02, M04 |
  | Moderate-severe | M05 |
  | Moderate | F03 |
  | Mild | F04, M03 |

  F01 has only about 228 clips.

## How it works (decided)

### Repo layout (the project folder becomes the repo)

```
clearvoice/            Python package, small modules
  data.py              load the dataset, parse IDs from paths, normalise text, group mic pairs, minutes per speaker
  splits.py            prompt holdout, speaker folds, random split, personalisation subsets; writes splits/*.csv
  asr.py               load a Whisper model; batch transcribe (language="en", task="transcribe", no timestamps)
  train.py             fine-tune whisper-small (Hugging Face Seq2SeqTrainer)
  metrics.py           WER/CER (jiwer), per-group tables, speaker-level bootstrap
  plots.py             figures
notebooks/
  01_data_and_splits.ipynb
  02_zero_shot_baselines.ipynb
  03_finetune.ipynb       run once per regime: fold A, fold B, random
  04_personalise.ipynb
  05_results_and_failures.ipynb
app/                   Gradio app and requirements.txt for the Hugging Face Space
splits/                committed CSVs of row indices per split
results/               committed prediction and metric CSVs, figures
docs/                  this design and the write-up draft
README.md              the write-up (Veritas allows a README) and setup steps
```

**Where things are stored.**
- Model checkpoints are saved to Google Drive during training.
- The final fold models are pushed to the user's Hugging Face account.
- Results CSVs and figures are committed to GitHub.

### Splits (all seeded and written to `splits/`)

1. **Prompt holdout.** Normalise transcriptions, then split the unique prompts 70% train, 10% validation and 20% test, stratified by single word versus sentence. Test and validation prompts never appear in any training set for any speaker.
2. **Speaker folds.** Each fold has two severe speakers, one mid and one mild:

   | Fold | Dysarthric speakers | Controls |
   |---|---|---|
   | A | F01, M01 (severe); F03 (moderate); M03 (mild) | FC01, MC01, MC02 |
   | B | M02, M04 (severe); M05 (moderate-severe); F04 (mild) | FC02, FC03, MC03, MC04 |

   The fold-A model trains on fold-B speakers (training prompts only), and the fold-B model trains on fold-A speakers. **Validation** is the training speakers saying the validation prompts. It is used to choose checkpoints, so checkpoint choice rewards new sentences, not memorised ones. Test data is never used to choose anything.
3. **Test sets per fold model**, on that fold's held-out speakers:
   - **Seen-sentence:** held-out speakers × training prompts.
   - **Strict:** held-out speakers × held-out prompts. This is the headline.
4. **Random split (stretch).** An 80/10/10 split over rows that deliberately ignores speakers and mic pairs. One fine-tune, used only for the leakage table's third row.
5. **Microphone.** Train on both microphones. Evaluate on one copy per utterance (prefer `headMic`, fall back to `arrayMic`) so no utterance is counted twice.

### Models

- **Baselines.** `openai/whisper-small` and `openai/whisper-large-v3`, zero-shot, same normaliser.
- **Fine-tune.** Full fine-tune of `openai/whisper-small` in bf16, trained on both dysarthric and control training speakers. The starting recipe is Hugging Face's "Fine-Tune Whisper" blog: learning rate about 1e-5, effective batch about 16–32, warmup about 10%, a few thousand steps, evaluation on validation WER, keep the best checkpoint, checkpoint to Drive regularly.
- **One recipe.** Whatever the strict split shows, small or negative gains included, is reported honestly. A second recipe is tried only if work is ahead of schedule, and is judged on validation only.
- **Personalisation.**
  - Take 2 held-out speakers (one severe, one mild or moderate); stretch to 4.
  - Starting from that fold's model, continue fine-tuning on k minutes of the speaker's own training-prompt clips, for k in {2, 5, 10} (stretch: 20), capped at what they have. Use a small learning rate and a fixed number of epochs.
  - Test on that speaker's strict clips. k = 0 is the fold model itself.
- **General speech.** If the fine-tuned model is clearly worse than stock whisper-small on the LibriSpeech check, report it and keep both outputs side by side in the demo. Mixing general speech into training is a stretch item.

### Metrics and reporting

- **Normalisation.** Normalise both references and model outputs with Whisper's `EnglishTextNormalizer` before scoring.
- **Metrics.** WER for sentences, CER for single words; reported overall, per named speaker, per severity and for controls.
- **Confidence intervals.** 95% intervals by bootstrap, resampling speakers for aggregate numbers.
- **Every table** shows the counts of clips and speakers behind each number.
- **Failure analysis.** The 10 worst strict-split clips by CER, each with a cause: severity, single-word confusion, hallucinated words, truncation, or label noise.

### Demo (Hugging Face Space, free CPU)

- **Tab "Unseen speakers":** about 20 pre-chosen strict-split clips across severities.
  - Each clip shows stock whisper-small, Clear Voice and the true text, plus per-clip error.
  - Each clip is run through the fold model that never heard that speaker.
  - Large-v3 outputs are precomputed (too heavy for CPU).
  - Clips are fetched from the Hugging Face dataset at runtime, not stored in the Space or the repo.
- **Tab "Try it":** microphone or upload, giving stock versus Clear Voice using the fold-A model. This is for the judges' typical-speech check.
- **Tab "Results":** the leakage table and personalisation curve images.
- **Disclaimer on every screen:** a research prototype, not a medical device; non-commercial research data.

### Submission

- **Repo:** a public GitHub repo under the user's account (`gh` is logged in as armaan-khoja), with notebooks committed with outputs.
- **README:** follows the Veritas write-up template. It includes the track-fit line ("Speech technology serves some speakers poorly...") and the "What we added" section (suggested project 4.4.3 plus our own user, data and questions).
- **Citations:** NP-TORGO, Project Euphonia, the Speech Accessibility Project and the TU Delft LoRA thesis.
- **Credits:** TORGO, the Hugging Face mirror, Whisper models, libraries, Claude.
- **Video:** 2 minutes, following the template (problem 15 s, demo 60 s, how it works 30 s, limitation 15 s).

### Work order (wall clock, ET)

| When | Done by the end |
|---|---|
| Sat 3–6 PM | Repo, data loaded in Colab, speaker IDs, mic pairs, splits committed |
| Sat 6–10 PM | Zero-shot baselines (small and large-v3); training code tested on a tiny run; fold A training started |
| Overnight | Fold A, then fold B training (tab left open) |
| Sun 8 AM–12 PM | Both fold models scored; general-speech check; Space v1 live |
| Sun 12–4 PM | Personalisation (2 speakers); failure analysis; figures |
| Sun 4–8 PM | README write-up, video, **first submission by 8 PM** |
| Sun 8–10 PM | Fixes, stretch items only if done early, **final submission by 10 PM**, tag `v1-submission` |

**If time runs short, cut in this order:**
1. Personalisation down to 1 speaker.
2. Demo tab "Results".
3. The general-speech check.

Never cut the strict split, the baselines or the per-speaker table.

## Appendix: Veritas requirements this build must meet

- **Write-up** (about one page, README accepted), in these sections: Project name, track and team; The problem; What we built; What we added (name the suggested project followed and what was added: own user, data or question); How it works (data, model); Results (baseline X versus model Y on held-out data, where it fails); What we'd do next; Sources.
- **Video** (optional, up to 2 minutes): the problem for about 15 s, the demo for about 60 s, how it works and the baseline comparison for about 30 s, one limitation and next step for about 15 s.
- **Every project answers:** who it is for, what goes in and out, whether it beats a simple baseline, and where it fails. All results come from unseen test data.
- **Suggested project 4.4.3, "Speech recognition across accents":** a few hundred clips per accent or language from Common Voice or FLEURS, transcribed with whisper-small, word error rate per group with jiwer. Its printed extension ideas include testing a fix such as fine-tuning on a small set of clips. Clear Voice follows its method and adds its own user, data and questions.
- **Languages track needs:** results per language or accent; model output shown beside the correct output; public data or the team's own voice. Judges also look for checks by a real speaker; for this project that means the speech therapist outreach.
- **Submission form:** code link (Colab or GitHub, viewable without sign-in, outputs visible), write-up link, track, team details, credits for every dataset, model, tutorial and AI tool.

## Left to the builder

- Exact hyperparameters and batch sizes.
- LoRA, if memory or time requires it.
- Gradient checkpointing.
- Code structure inside modules.
- Plot styling.
- Which specific speakers to personalise, within the rule above.
- Which ~20 demo clips to use (chosen from the strict split only, spread across severities).
- Notebook resume logic.

# Clear Voice

Track: Languages. Team: TODO(team names).

> Research prototype, not a medical device. It uses the TORGO database, which is licensed for non-commercial research only.

## The problem

Dysarthria makes speech slow, slurred or quiet, because the muscles used for speaking are weak or hard to control. It comes with cerebral palsy, ALS, stroke and Parkinson's disease. Phone dictation and live captions are trained mostly on typical voices, so they turn this speech into the wrong words, and a tool everyone else uses to text a friend or follow a lecture stops working for the people who would gain most from it.

## What we built

Clear Voice is a speech recogniser fine-tuned on dysarthric speech, together with a test that tries hard not to flatter it. You give it a recording and it returns the words. The person we had in mind is a teenager with cerebral palsy dictating a message, and the speech therapist deciding whether a few minutes of that teenager's own recordings would make the tool work for them.

The demo is a small web app. One tab plays recordings from people the model never heard and shows what stock Whisper and Clear Voice each wrote next to what was actually said. Another lets anyone record themselves.

We started from suggested project 4.4.3, "Speech recognition across accents", which fine-tunes Whisper and compares word error rates across groups. We changed three things. The users are people with dysarthria, who are served far worse than accented speakers. The data is TORGO instead of Common Voice. And the test asks harder questions: does the model work for a new person saying a new sentence, how much do easier tests inflate the score, how many minutes of someone's own voice does personalisation need, and did fine-tuning damage ordinary speech?

## How it works

TORGO (we used the Hugging Face mirror `abnerh/TORGO-database`) has 16,552 recordings from 15 speakers: 8 with dysarthria from cerebral palsy or ALS, ranging from mild to severe, and 7 typical speakers. Everyone reads the same list of about 960 prompts, some single words and some sentences, and every utterance was captured on two microphones. We train on both copies but score only one, so nothing is counted twice, and we dropped two recordings longer than Whisper's 30-second window.

Because everyone reads the same prompts, a random split lets a model memorise sentences it will later be tested on. Earlier work found the same problem in TORGO (the NP-TORGO split). We split twice. The prompts are divided 70% training, 10% validation and 20% test, separately for words and sentences, and near-identical prompts such as "as I choose" and "as I chose" are kept on the same side. The 8 dysarthric speakers are divided into two groups of four, each with two severe, one moderate and one mild speaker (A: F01, M01, F03, M03; B: M02, M04, M05, F04). We train one model per group and test it on the other group's speakers, using only test prompts. So every number in the headline comes from a person the model never heard saying a sentence it never saw. The checkpoint is chosen on validation prompts, so choosing it never rewards memorisation either.

The model is OpenAI's whisper-small fine-tuned with learning rate 1e-5 and batch size 16 on one L4 GPU. Validation word error rate stopped improving after about 1,000 steps, and early stopping kept the best checkpoint. We compare it with stock whisper-small, the model it started from, and stock whisper-large-v3, which is about six times bigger.

We normalise case, punctuation and numbers on both sides before scoring. Sentences are scored by word error rate and single words by character error rate. The 95% intervals come from resampling whole speakers, because with eight speakers the variation between people matters more than the variation between clips. We also report the median clip, because Whisper sometimes produces a long run of invented words for a single short recording, and one such clip can move a total a long way.

## Results

TODO(final): round 2 (whisper-large-v3-turbo as the base, tuned personalisation) is running; the model reported here will be the one with the best validation score, and both will be listed.

New speakers with dysarthria, saying prompts the models never saw (690 recordings, 8 speakers):

| System | Sentences: words wrong | 95% interval | Median clip | Single words: characters wrong | 95% interval |
|---|---|---|---|---|---|
| Stock whisper-small | 44.7% | 16% to 87% | 16.7% | 66.8% | 42% to 98% |
| Stock whisper-large-v3 | 22.1% | 11% to 37% | 8.3% | 57.4% | 35% to 84% |
| Clear Voice (fine-tuned whisper-small) | 31.8% | 17% to 50% | 25.0% | 41.0% | 30% to 52% |

Clear Voice is the best of the three on single words, and it cuts sentence errors by 29% compared with the model it started from. It does not beat the much larger stock model on sentences. The mean and the median also disagree: stock whisper-small's mean is pulled up by a few recordings where it invented long strings of words, while its typical sentence was better than Clear Voice's. Fine-tuning seems to have removed most of those runaway outputs at the cost of more small mistakes on sentences stock Whisper already got nearly right. TODO(check): confirm this from the per-clip predictions.

The same Clear Voice model looks better on easier tests:

| Test (dysarthric speakers only) | Sentences: words wrong | Single words: characters wrong |
|---|---|---|
| New speaker, prompts seen in training | 19.0% | 19.7% |
| Seen speakers, new prompts | 26.1% | 34.0% |
| New speaker, new prompts (headline) | 31.8% | 41.0% |

Had we tested on prompts the model had already practised, we would have reported 19% instead of 32% on sentences, and 20% instead of 41% on single words. The second row uses the other group's speakers, so it is not a like-for-like comparison with the first and third.

Fine-tuning barely touched ordinary speech. On 200 LibriSpeech recordings, Clear Voice gets 3.6% to 4.1% of words wrong (one model per group) against 3.3% for stock whisper-small and 1.4% for whisper-large-v3.

Personalisation: our first recipe adapted the model to one speaker with 2, 5 and 10 minutes of that person's recordings, at learning rate 1e-5 for 5 epochs. It barely helped. For M01 (severe) errors went from 58% to 51% with 10 minutes, and for F03 (moderate) they stayed at 32%. That recipe was too cautious, so we tuned it on each speaker's other recordings (never the test prompts) and reran the curve. TODO(final): tuned curve.

TODO(final): where it fails, from listening to the 10 worst recordings.

## Limits

TORGO has only 8 speakers with dysarthria, so each model is trained on four of them; the intervals above are wide for that reason. The recordings are read prompts in quiet rooms, not real dictation. The data is licensed for non-commercial research, so this is a prototype, not something that could be deployed. And nobody with dysarthria has used Clear Voice yet.

## What we would do next

Test it with a speech therapist and with people with dysarthria who agree to take part, on their own dictation instead of read prompts. Train on a larger collection gathered with consent, such as the Speech Accessibility Project, so each model sees far more than four dysarthric speakers.

## Sources

Data:
- Rudzicz, F., Namasivayam, A. K. and Wolff, T. (2012). The TORGO database of acoustic and articulatory speech from speakers with dysarthria. Language Resources and Evaluation 46(4). Mirror: https://huggingface.co/datasets/abnerh/TORGO-database
- Panayotov, V. et al. (2015). LibriSpeech: an ASR corpus based on public domain audio books. https://huggingface.co/datasets/openslr/librispeech_asr

Models: OpenAI Whisper (Radford et al., 2023, "Robust Speech Recognition via Large-Scale Weak Supervision"): `openai/whisper-small`, `openai/whisper-large-v3`, `openai/whisper-tiny` (tests only).

Related work:
- Hui, M., Zhang, J. and Mohan, A. (2024). Enhancing AAC Software for Dysarthric Speakers in e-Health Settings: An Evaluation Using TORGO. arXiv:2411.00980. They found TORGO's shared prompts inflate results and built NP-TORGO, a split with no prompt overlap; our strict split follows the same idea.
- Google's Project Euphonia.
- The Speech Accessibility Project (University of Illinois).
- A TU Delft thesis on fine-tuning Whisper with LoRA on TORGO.

Tutorial: Hugging Face, "Fine-Tune Whisper For Multilingual ASR with Transformers".

Libraries: PyTorch, Hugging Face Transformers, huggingface_hub, jiwer, pandas, PyArrow, SciPy, soundfile, Matplotlib, Gradio, pytest; Remotion for the demo video.

AI tools: Claude (Anthropic) helped us brainstorm the idea, write and test the code, design the video and draft this write-up. We reviewed every part.

## Reproduce

Open the notebooks in Google Colab with an L4 GPU and run `notebooks/01_data_and_splits`, `02_zero_shot_baselines`, `03_finetune` (once with `HELDOUT = "A"` and once with `"B"`), `04_personalise` and `05_results_and_failures`. Results and models are saved to Google Drive under `MyDrive/clearvoice`. `scripts/overnight.py` runs the same steps one after another in a single Colab session.

To run the tests locally:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

```bash
.venv/bin/pytest
```

`.venv/bin/pytest -m slow` also runs the slower tests, which download whisper-tiny. `splits/` holds the row indices of every split; `scripts/make_splits.py` rebuilds them, and notebook 1 checks that the rebuild matches.

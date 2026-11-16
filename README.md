# Clear Voice

Track: Languages. Team: Armaan Khoja and Suvir Aylawadi.

**Demo video:** https://armaan-khoja.github.io/clear-voice/  
**Research paper:** [Research Paper Updated Final.pdf](Research%20Paper%20Updated%20Final.pdf)

> Research prototype, not a medical device. It uses the TORGO database, which is licensed for non-commercial research only.

Copyright © 2026 Armaan Khoja and Suvir Aylawadi. All rights reserved. See [LICENSE](LICENSE).

## The problem

Dysarthria makes speech slow, slurred or quiet, because the muscles used for speaking are weak or hard to control. It comes with cerebral palsy, ALS, stroke and Parkinson's disease. Phone dictation and live captions are trained mostly on typical voices, so they turn this speech into the wrong words, and a tool everyone else uses to text a friend or follow a lecture stops working for the people who would gain most from it.

## What we built

Clear Voice is a speech recogniser fine-tuned on dysarthric speech, together with a test that tries hard not to flatter it. You give it a recording and it returns the words. The person we had in mind is a teenager with cerebral palsy dictating a message, and the speech therapist deciding whether a few minutes of that teenager's own recordings would make the tool work for them.

The demo is a video, plus a small web app in `app/` built to run as a Hugging Face Space. One tab plays recordings from people the model never heard and shows what stock Whisper and Clear Voice each wrote next to what was actually said. Another lets anyone record themselves. The Space is not online yet, because the fine-tuned model has not been uploaded.

We started from suggested project 4.4.3, "Speech recognition across accents", which fine-tunes Whisper and compares word error rates across groups. We changed three things. The users are people with dysarthria, who are served far worse than accented speakers. The data is TORGO instead of Common Voice. And the test asks harder questions: does the model work for a new person saying a new sentence, how much do easier tests inflate the score, how many minutes of someone's own voice does personalisation need, and did fine-tuning damage ordinary speech?

## How it works

TORGO (we used the Hugging Face mirror `abnerh/TORGO-database`) has 16,552 recordings from 15 speakers: 8 with dysarthria from cerebral palsy or ALS, ranging from mild to severe, and 7 typical speakers. Everyone reads the same list of about 960 prompts, some single words and some sentences, and every utterance was captured on two microphones. We train on both copies but score only one, so nothing is counted twice, and we dropped two recordings longer than Whisper's 30-second window.

Because everyone reads the same prompts, a random split lets a model memorise sentences it will later be tested on. Earlier work found the same problem in TORGO (the NP-TORGO split). We split twice. The prompts are divided 70% training, 10% validation and 20% test, separately for words and sentences, and near-identical prompts such as "as I choose" and "as I chose" are kept on the same side. The 8 dysarthric speakers are divided into two groups of four, each with two severe, one moderate and one mild speaker (A: F01, M01, F03, M03; B: M02, M04, M05, F04). We train one model per group and test it on the other group's speakers, using only test prompts. So every number in the headline comes from a person the model never heard saying a sentence it never saw. The checkpoint is chosen on validation prompts, so choosing it never rewards memorisation either.

We fine-tuned three versions of OpenAI's Whisper, whisper-small (244 million parameters), whisper-medium (769 million) and whisper-large-v3-turbo (809 million), with learning rate 1e-5 and an effective batch of 16 on one L4 GPU, scoring the validation sentences every 250 steps and keeping the best checkpoint. We chose between them on validation alone: averaged over both groups, fine-tuned whisper-medium got 15.1% of validation words wrong, large-v3-turbo 17.0% and whisper-small 17.2%, so Clear Voice is the fine-tuned whisper-medium. We compare it with the stock model it started from, stock whisper-small, and stock whisper-large-v3, which is twice the size of whisper-medium.

We normalise case, punctuation and numbers on both sides before scoring. Sentences are scored by word error rate and single words by character error rate. The 95% intervals come from resampling whole speakers, because with eight speakers the variation between people matters more than the variation between clips. We also report the median clip, because Whisper sometimes produces a long run of invented words for a single short recording, and one such clip can move a total a long way.

## Results

New speakers with dysarthria, saying prompts the models never saw (690 recordings, 8 speakers):

| System | Sentences: words wrong | 95% interval | Median clip | Single words: characters wrong | 95% interval |
|---|---|---|---|---|---|
| Stock whisper-small | 44.7% | 16% to 87% | 16.7% | 66.8% | 42% to 98% |
| Stock whisper-medium (what Clear Voice started from) | 33.8% | 14% to 62% | 14.3% | 95.8% | 50% to 146% |
| Stock whisper-large-v3 | 22.1% | 11% to 37% | 8.3% | 57.4% | 35% to 84% |
| **Clear Voice (fine-tuned whisper-medium)** | **29.2%** | 14% to 49% | 12.5% | **39.3%** | 29% to 50% |

Clear Voice is the best of the four on single words, by a wide margin, and it gets 14% fewer sentence words wrong than the model it started from. It does not beat stock whisper-large-v3 on sentences.

The single-word result is mostly about invented text. Asked to transcribe one short word from a speaker with dysarthria, stock Whisper often writes something long instead: for the word "brawn" from speaker M05, stock whisper-medium wrote a "B" followed by a long run of the letter U. Character error rates can pass 100% for that reason, as stock whisper-medium's interval does. Sentences show the same pattern less dramatically. For M04, who has severe dysarthria, stock whisper-small's word error rate on sentences was 157%: it wrote more wrong words than there were words. Fine-tuned whisper-small brought that down to 67%. The cost of fine-tuning is more small mistakes on sentences that stock Whisper already got right. In a random sample of test recordings, stock Whisper heard "I feel I can play this weekend" correctly from M04, and Clear Voice wrote "I feel high-spin this weekend".

Every model we tried, on the same test:

| System | Validation: words wrong | Sentences: words wrong | Single words: characters wrong |
|---|---|---|---|
| Stock whisper-small | | 44.7% | 66.8% |
| Stock whisper-medium | | 33.8% | 95.8% |
| Stock whisper-large-v3-turbo | | 24.8% | 64.9% |
| Stock whisper-large-v3 | | 22.1% | 57.4% |
| Fine-tuned whisper-small | 17.2% | 31.8% | 41.0% |
| Fine-tuned whisper-medium (Clear Voice) | 15.1% | 29.2% | 39.3% |
| Fine-tuned whisper-large-v3-turbo | 17.0% | 41.1% | 41.0% |

Fine-tuned large-v3-turbo shows why the choice is made on validation. It scored behind whisper-medium there, so it was never in the running, and on the test it was worse than the stock model it started from: 41.1% of sentence words wrong against 24.8%, with its typical sentence twice as bad (median 28.6% against 14.3%). In one recording it got stuck repeating "a different way for" until it hit the length limit. Like the others, it did help on single words.

### Adding Say It's sentence corrector

Say It is a browser app Suvir built for the project. It runs stock Whisper on the person's own device, then corrects the text. It lists sentences the speaker might have meant: similar-sounding words, short words like "the" and "to" put back, and everyday phrases from a list of 215. Then 300 gradient-boosted decision trees pick one, keeping Whisper's text unless they are clearly sure. The corrector works on text, so it can run on Clear Voice's output too. We ran it exactly as the app does and decided on validation, as for the model choice.

| System | Validation: words wrong | Sentences: words wrong | Single words: characters wrong |
|---|---|---|---|
| Clear Voice | 15.1% | 29.2% | 39.3% |
| Clear Voice + Say It corrector | 33.4% | 30.2% | 81.1% |
| Stock whisper-large-v3 | | 22.1% | 57.4% |
| Stock whisper-large-v3 + Say It corrector | | 22.7% | 76.6% |

It more than doubled the validation errors, so Clear Voice does not use it. On the strict test it changed 207 of the 690 dysarthric clips: 3 for the better and 173 for the worse. Nearly all the damage is on single words, which it turns into everyday phrases from its list: "feet" became "my feet hurt" and "witty" became "maybe". On sentences it made little difference either way. We also tried lighter versions: using it only when the model heard two or more words, and without the phrase list. They stop the damage to single words but do not help: the best on validation was still Clear Voice alone (15.1% of words wrong, against 15.2% for the sentences-only version), and on the test the sentences-only version fixed 3 clips and broke 9. This is not the job the corrector was built for, which is a person saying their own everyday requests, and TORGO's word lists and read sentences are not that. Say It's other feature, recognising a person's saved phrases from their voice, is not tested here. To combine them properly we would retrain the trees on Clear Voice's output, use them only on sentences, and limit the phrase list to the person's own phrases.

The same Clear Voice model looks better on easier tests:

| Test (dysarthric speakers only) | Sentences: words wrong | Single words: characters wrong |
|---|---|---|
| New speaker, prompts seen in training | 15.5% | 18.7% |
| Seen speakers, new prompts | 18.9% | 32.6% |
| New speaker, new prompts (headline) | 29.2% | 39.3% |

Had we tested on prompts the model had already practised, we would have reported 16% instead of 29% on sentences, and 19% instead of 39% on single words. The second row uses the other group's speakers, so it is not a like-for-like comparison with the first and third.

Fine-tuning cost a little on ordinary speech. On 200 LibriSpeech recordings, Clear Voice gets 3.0% of words wrong with either group's model, against 2.4% for stock whisper-medium, 3.3% for stock whisper-small and 1.4% for whisper-large-v3.

Personalisation: we adapted the fine-tuned whisper-small to one speaker with 2, 5 and 10 minutes of that person's own recordings. To pick the training recipe without touching the test, we tried four recipes on each speaker's recordings of validation sentences. The most cautious one (learning rate 1e-5, 5 epochs) won for both speakers; training longer or harder made new sentences worse, for example from 20% to 44% of words wrong for F03 at learning rate 5e-5. With that recipe, 10 minutes of M01's voice (severe) brought errors on new sentences from 58% to 52%, and F03's (moderate) stayed at 32%. Ten minutes of read prompts is not enough to adapt to a voice without memorising the prompts.

Where it fails, from reading the transcripts of the worst recordings and of a random sample: most failures are single words that come back as a different real word ("eight" as "root", "ate" as "hitting", "pit" as "beach"), and occasionally a repeated word ("briar" as "The best best best best"). On sentences, the errors are near-misses on the unusual words, such as "The prospects of cupping" as "The prospect of capital", and changes to sentences stock Whisper already had right.

## Limits

TORGO has only 8 speakers with dysarthria, so each model is trained on four of them; the intervals above are wide for that reason. The recordings are read prompts in quiet rooms, not real dictation. The data is licensed for non-commercial research, so this is a prototype, not something that could be deployed. And nobody with dysarthria has used Clear Voice yet.

## What we would do next

Test it with a speech therapist and with people with dysarthria who agree to take part, on their own dictation instead of read prompts. Train on a larger collection gathered with consent, such as the Speech Accessibility Project, so each model sees far more than four dysarthric speakers. Put Clear Voice inside Say It, Suvir's browser app that remembers a person's own phrases by their voice, with its corrector retrained on Clear Voice's output and used only on sentences.

## Sources

Data:
- Rudzicz, F., Namasivayam, A. K. and Wolff, T. (2012). The TORGO database of acoustic and articulatory speech from speakers with dysarthria. Language Resources and Evaluation 46(4). Mirror: https://huggingface.co/datasets/abnerh/TORGO-database
- Panayotov, V. et al. (2015). LibriSpeech: an ASR corpus based on public domain audio books. https://huggingface.co/datasets/openslr/librispeech_asr

Models: OpenAI Whisper (Radford et al., 2023, "Robust Speech Recognition via Large-Scale Weak Supervision", https://arxiv.org/abs/2212.04356): [whisper-small](https://huggingface.co/openai/whisper-small), [whisper-medium](https://huggingface.co/openai/whisper-medium), [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3), [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo), [whisper-tiny](https://huggingface.co/openai/whisper-tiny) (tests only).

Related work:
- Hui, M., Zhang, J. and Mohan, A. (2024). Enhancing AAC Software for Dysarthric Speakers in e-Health Settings: An Evaluation Using TORGO. arXiv:2411.00980. They found TORGO's shared prompts inflate results and built NP-TORGO, a split with no prompt overlap; our strict split follows the same idea.
- Google's Project Euphonia.
- The Speech Accessibility Project (University of Illinois).
- A TU Delft thesis on fine-tuning Whisper with LoRA on TORGO.

Tutorial: Hugging Face, "Fine-Tune Whisper For Multilingual ASR with Transformers", https://huggingface.co/blog/fine-tune-whisper

Libraries: PyTorch, Hugging Face Transformers, huggingface_hub, jiwer, pandas, PyArrow, SciPy, soundfile, Matplotlib, Gradio, pytest; Remotion for the demo video.

Say It (Suvir's companion app, whose corrector we tested): runs stock Whisper in the browser with [Transformers.js](https://huggingface.co/docs/transformers.js) ([onnx-community/whisper-large-v3-turbo](https://huggingface.co/onnx-community/whisper-large-v3-turbo), [whisper-small.en](https://huggingface.co/onnx-community/whisper-small.en), [whisper-base.en](https://huggingface.co/onnx-community/whisper-base.en)) and [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/), recognises saved phrases with the whisper-base.en encoder from [sherpa-onnx](https://huggingface.co/csukuangfj/sherpa-onnx-whisper-base.en), and picks corrections with gradient-boosted decision trees ([XGBoost](https://xgboost.ai); Friedman, 2001).

AI tools: [Claude](https://www.anthropic.com/claude) (Anthropic) helped us write and test the code, design the video and draft this write-up. We reviewed every part.

## Reproduce

Open the notebooks in Google Colab with an L4 GPU and run `notebooks/01_data_and_splits`, `02_zero_shot_baselines`, `03_finetune` (once with `HELDOUT = "A"` and once with `"B"`), `04_personalise` and `05_results_and_failures`. Results and models are saved to Google Drive under `MyDrive/clearvoice`. `scripts/overnight.py` runs the same steps one after another in a single Colab session.

To run the tests locally:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

```bash
.venv/bin/pytest
```

`scripts/sayit/evaluate.py` reruns the Say It corrector test from the app's HTML file and the saved predictions, and `scripts/sayit/variants.js` runs the lighter versions on the same inputs; both need Node.js.

`.venv/bin/pytest -m slow` also runs the slower tests, which download whisper-tiny. `splits/` holds the row indices of every split; `scripts/make_splits.py` rebuilds them, and notebook 1 checks that the rebuild matches.

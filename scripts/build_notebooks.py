"""Generate the five Colab notebooks in notebooks/. Run: python scripts/build_notebooks.py"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = '''# Setup: run this cell first. On Colab it clones the repo, installs packages and mounts Google Drive.
REPO_URL = "https://github.com/armaan-khoja/clear-voice"
SMOKE = False  # True = a quick check on a handful of clips (minutes, not hours)

import os, subprocess, sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not os.path.exists("/content/clear-voice"):
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/clear-voice"], check=True)
    os.chdir("/content/clear-voice")
    subprocess.run(["git", "pull", "-q"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
    from google.colab import drive, userdata
    drive.mount("/content/drive")
    WORK = "/content/drive/MyDrive/clearvoice"
    try:
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    except Exception:
        print("No HF_TOKEN secret: downloads still work, uploading models will not.")
else:
    WORK = os.path.abspath("work")
if SMOKE:
    WORK = f"{WORK}/smoke"  # quick checks never overwrite real models or results
sys.path.insert(0, os.getcwd())
os.makedirs(f"{WORK}/results", exist_ok=True)
os.makedirs(f"{WORK}/models", exist_ok=True)
SPLITS = "splits"
LIMIT = 6 if SMOKE else None
print("Results and models go to", WORK)'''

GPU = '''import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "No GPU: Runtime > Change runtime type > L4 GPU, then rerun the setup cell.")'''

LOAD = '''from clearvoice.pipeline import load_torgo
parquets, meta = load_torgo(WORK)  # downloads TORGO (about 1.6 GB) the first time in each session
print(len(meta), "clips from", meta.speaker.nunique(), "speakers")'''


def nb(title: str, intro: str, cells: list) -> nbformat.NotebookNode:
    book = new_notebook()
    book.cells = [new_markdown_cell(f"# {title}\n\n{intro}"), new_code_cell(SETUP), new_code_cell(GPU)]
    for kind, text in cells:
        book.cells.append(new_markdown_cell(text) if kind == "md" else new_code_cell(text))
    book.metadata["accelerator"] = "GPU"
    book.metadata["colab"] = {"provenance": [], "gpuType": "L4"}
    book.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    return book


BOOKS = {
    "01_data_and_splits": nb(
        "1. Data and splits",
        "Loads TORGO, checks it, and confirms that the splits saved in the repo rebuild exactly. "
        "No GPU needed. Takes about 3 minutes.",
        [
            ("code", LOAD),
            ("md", "**Minutes of speech per speaker** (both microphone copies counted)."),
            ("code", 'meta.groupby(["speaker", "severity"]).duration.sum().div(60).round(1)'),
            ("md", "**Data checks.** Every utterance was recorded on two microphones; both copies must say the same thing."),
            ("code", '''from clearvoice.data import check_mic_pairs
print("Utterances whose two microphone copies disagree:", check_mic_pairs(meta).utt_key.nunique())
print("Clips longer than 30 s (dropped from every split):", (meta.duration > 30).sum())'''),
            ("md", "**Rebuild the splits** and check they match the CSVs in `splits/`. "
                   "Test and validation sentences never appear in training; held-out speakers never appear in training."),
            ("code", '''from clearvoice.splits import assign_prompt_splits, fold_splits, read_split
ps = assign_prompt_splits(meta)
for held in ["A", "B"]:
    for part, df in fold_splits(meta, ps, held).items():
        saved = read_split(SPLITS, f"hold{held}", part)
        assert list(saved.row_id) == list(df.row_id), (held, part)
print("Splits in the repo match a fresh rebuild.")'''),
        ],
    ),
    "02_zero_shot_baselines": nb(
        "2. Zero-shot baselines",
        "Stock whisper-small and whisper-large-v3, with no fine-tuning, on every speaker saying unseen sentences, "
        "plus 200 clips of everyday read English (LibriSpeech). About 30-45 minutes on an L4.",
        [
            ("code", LOAD),
            ("code", '''import pandas as pd, torch
from clearvoice.asr import load_model
from clearvoice.pipeline import librispeech_check, predict_fold

general = []
for name in ["openai/whisper-small", "openai/whisper-large-v3"]:
    short = name.split("/")[1]
    model, processor = load_model(name)
    preds = pd.concat([predict_fold(model, processor, parquets, meta, SPLITS, fold, ["test_strict"], short,
                                    batch_size=32, limit=LIMIT) for fold in ["A", "B"]], ignore_index=True)
    preds.to_csv(f"{WORK}/results/preds_zeroshot_{short}.csv", index=False)
    ls = librispeech_check(model, processor, n=20 if SMOKE else 200, batch_size=32)
    general.append({"system": short, "librispeech_wer": ls["wer"], "n": ls["n"]})
    print(short, "done")
    del model
    torch.cuda.empty_cache()
pd.DataFrame(general).to_csv(f"{WORK}/results/librispeech_zeroshot.csv", index=False)
pd.DataFrame(general)'''),
            ("md", "**First look:** dysarthric speakers, unseen sentences. Sentences scored by word error rate, "
                   "single words by character error rate."),
            ("code", '''from clearvoice.pipeline import read_preds
from clearvoice.report import headline
headline({s: read_preds(f"{WORK}/results/preds_zeroshot_{s}.csv") for s in ["whisper-small", "whisper-large-v3"]})'''),
        ],
    ),
    "03_finetune": nb(
        "3. Fine-tune Clear Voice",
        "Fine-tunes whisper-small on one fold's speakers and tests it on the other fold's speakers. "
        "**Run it twice: once with `HELDOUT = \"A\"`, then with `HELDOUT = \"B\"`.** About 60-90 minutes each on an L4. "
        "Keep this browser tab open while it trains.",
        [
            ("code", '''HELDOUT = "A"        # run once with "A", then once with "B"
PUSH_TO_HUB = True   # uploads the model to your Hugging Face account for the demo (needs the HF_TOKEN secret)
RETRAIN = False      # False: if this fold's model is already on Drive, skip straight to testing'''),
            ("code", LOAD),
            ("code", '''from clearvoice.data import load_audio
from clearvoice.pipeline import rows_for
name = f"hold{HELDOUT}"
train = rows_for(meta, SPLITS, name, "train")
val = rows_for(meta, SPLITS, name, "val")
if SMOKE:
    train, val = train.head(16), val.head(8)
print(len(train), "training clips from", sorted(train.speaker.unique()))
print(len(val), "validation clips (validation sentences only)")
from clearvoice.pipeline import kept_model
needs_training = RETRAIN or kept_model(WORK, name) is None
audio = load_audio(parquets, meta, list(train.row_id) + list(val.row_id)) if needs_training else {}'''),
            ("md", "**Train.** Learning rate 1e-5, batch 16, up to 2,000 steps. Every 250 steps the model is scored on "
                   "validation sentences it never trains on; the best one is kept, and training stops early if it stops improving."),
            ("code", '''from clearvoice.pipeline import kept_model, save_best
from clearvoice.train import TrainConfig, finetune
best = kept_model(WORK, name)
if best and not RETRAIN:
    print("Using the model already trained:", best)
else:
    cfg = TrainConfig()
    if SMOKE:
        cfg = TrainConfig(max_steps=4, eval_steps=2, save_steps=2, batch_size=4, eval_batch_size=4)
    # Train on Colab's own disk: checkpoints are ~1 GB each and would fill Google Drive (deleted files go to its trash).
    local = f"/content/models/{name}" if IN_COLAB else f"{WORK}/local_models/{name}"
    best = save_best(finetune(cfg, train, val, audio, local), WORK, name)
    print("Kept model saved to:", best)
del audio'''),
            ("md", "**Test** on the held-out speakers (seen and unseen sentences), on the training speakers' unseen "
                   "sentences, and on everyday English."),
            ("code", '''import pandas as pd
from clearvoice.asr import load_model
from clearvoice.pipeline import librispeech_check, predict_fold
from clearvoice.report import leakage
model, processor = load_model(best)
preds = predict_fold(model, processor, parquets, meta, SPLITS, HELDOUT,
                     ["test_seen", "test_newsent", "test_strict"], "Clear Voice", batch_size=32, limit=LIMIT)
preds.to_csv(f"{WORK}/results/preds_ft_{name}.csv", index=False)
ls = librispeech_check(model, processor, n=20 if SMOKE else 200, batch_size=32)
pd.DataFrame([{"system": "Clear Voice", "fold": HELDOUT, "librispeech_wer": ls["wer"], "n": ls["n"]}]).to_csv(
    f"{WORK}/results/librispeech_ft_{name}.csv", index=False)
leakage(preds)'''),
            ("code", '''if PUSH_TO_HUB and not SMOKE:  # never upload a smoke-test model
    from huggingface_hub import whoami
    repo = f"{whoami()['name']}/clear-voice-{name.lower()}"
    model.push_to_hub(repo)
    processor.push_to_hub(repo)
    print("Uploaded to https://huggingface.co/" + repo)'''),
        ],
    ),
    "04_personalise": nb(
        "4. Personalisation",
        "Starting from the fold-A model, adds 2, 5 and 10 minutes of one speaker's own recordings and tests on that "
        "speaker's unseen sentences. Needs notebook 3 run with `HELDOUT = \"A\"` first. About 30 minutes on an L4.",
        [
            ("code", '''SPEAKERS = ["M01", "F03"]   # stretch: add "M03" (fold A) and "M04" (needs the fold-B model)
MINUTES = [2, 5, 10]
FOLD_OF = {"M01": "A", "F03": "A", "M03": "A", "M04": "B"}'''),
            ("code", LOAD),
            ("code", '''import os, shutil, tempfile
import pandas as pd, torch
from clearvoice.asr import load_model
from clearvoice.data import load_audio
from clearvoice.metrics import score
from clearvoice.pipeline import predict, rows_for
from clearvoice.train import TrainConfig, finetune

def evaluate(model_dir, test, test_audio, fold, minutes):
    model, processor = load_model(model_dir)
    p = predict(model, processor, parquets, meta, test, "Clear Voice", fold, "test_strict", audio=test_audio)
    s = score(list(p.ref), list(p.hyp))
    del model
    torch.cuda.empty_cache()
    return {"speaker": test.speaker.iloc[0], "severity": test.severity.iloc[0], "minutes": minutes,
            "wer": s["wer"], "cer": s["cer"], "n_clips": s["n"]}

results = []
for spk in SPEAKERS:
    fold = FOLD_OF[spk]
    base = f"{WORK}/models/hold{fold}/best"
    test = rows_for(meta, SPLITS, f"hold{fold}", "test_strict")
    test = test[test.speaker == spk].head(LIMIT or len(test))
    test_audio = load_audio(parquets, meta, test.row_id)
    results.append(evaluate(base, test, test_audio, fold, 0))
    for k in MINUTES:
        path = f"{SPLITS}/personal_{spk}_{k}min.csv"
        if not os.path.exists(path):
            print(f"{spk}: {k} min is more than this speaker has (capped)")
            continue
        adapt = rows_for(meta, SPLITS, f"personal_{spk}", f"{k}min")
        adapt = adapt.head(8) if SMOKE else adapt
        audio = load_audio(parquets, meta, adapt.row_id)
        cfg = TrainConfig(base_model=base, max_steps=2 if SMOKE else None, num_train_epochs=5, eval=False, batch_size=8)
        tmp = tempfile.mkdtemp()  # personal models stay off Google Drive (about 1 GB each)
        out = finetune(cfg, adapt, None, audio, tmp)
        results.append(evaluate(out, test, test_audio, fold, k))
        shutil.rmtree(tmp, ignore_errors=True)
        print(spk, k, "min done")

res = pd.DataFrame(results)
res.to_csv(f"{WORK}/results/personalise.csv", index=False)
res'''),
        ],
    ),
    "05_results_and_failures": nb(
        "5. Results, figures and failures",
        "Builds every table and figure for the write-up and picks the demo clips. No GPU needed. A few minutes.",
        [
            ("code", '''import glob, os
import pandas as pd
from IPython.display import Image, display
from clearvoice import plots
from clearvoice.pipeline import align_folds, read_preds
from clearvoice.report import headline, leakage, per_speaker, pick_demo_clips, worst_clips
R = f"{WORK}/results"
zs = {s: read_preds(f"{R}/preds_zeroshot_{s}.csv") for s in ["whisper-small", "whisper-large-v3"]}
ft = pd.concat([read_preds(f"{R}/preds_ft_hold{f}.csv") for f in ["A", "B"]
                if os.path.exists(f"{R}/preds_ft_hold{f}.csv")], ignore_index=True)
zs = align_folds(zs, ft)  # same speakers for every system
preds = {**zs, "Clear Voice": ft}
print("Clear Voice folds found:", sorted(ft.fold.unique()))'''),
            ("md", "**Headline:** dysarthric speakers the model never heard, saying sentences it never heard (95% intervals from resampling speakers)."),
            ("code", "h = headline(preds)\nh"),
            ("md", "**Leakage table:** the same model gets better-looking scores on easier tests."),
            ("code", "lk = leakage(ft)\nlk"),
            ("md", "**Per speaker** (unseen sentences), including control speakers."),
            ("code", "ps = per_speaker(preds)\nps.pivot_table(index=['speaker', 'severity'], columns='system', values='word_cer').round(3)"),
            ("md", "**Everyday English** (LibriSpeech test-clean, 200 clips): did fine-tuning hurt typical speech?"),
            ("code", 'general = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{R}/librispeech_*.csv"))], ignore_index=True)\ngeneral'),
            ("md", "**The 10 worst clips.** Listen to these and write down why each one failed."),
            ("code", 'w = worst_clips(ft, n=10)\nw[["row_id", "speaker", "severity", "prompt_type", "ref", "hyp", "cer"]]'),
            ("md", "**Save tables, figures and demo clips.**"),
            ("code", '''os.makedirs(f"{R}/figures", exist_ok=True)
plots.bar_systems(h, f"{R}/figures/headline.png")
plots.leakage(lk, f"{R}/figures/leakage.png")
plots.per_speaker(ps, f"{R}/figures/per_speaker.png")
if os.path.exists(f"{R}/personalise.csv"):
    plots.personalisation(pd.read_csv(f"{R}/personalise.csv"), f"{R}/figures/personalisation.png")
for name, df in {"headline": h, "leakage": lk, "per_speaker": ps, "worst_clips": w, "general_speech": general}.items():
    df.to_csv(f"{R}/{name}.csv", index=False)

demo = pick_demo_clips(ft, per_severity=5).rename(columns={"hyp": "Clear Voice"})
for s, df in zs.items():
    demo = demo.merge(df[["row_id", "hyp"]].rename(columns={"hyp": s}), on="row_id", how="left")
meta = pd.read_pickle(f"{WORK}/meta.pkl")  # saved by notebook 1: where each clip sits in the dataset files
demo = demo.merge(meta[["row_id", "file_idx", "row_in_file"]], on="row_id", how="left")
demo.to_csv(f"{R}/demo_clips.csv", index=False)

with open(f"{R}/summary.md", "w") as f:
    for name, df in {"Headline": h, "Leakage": lk, "Everyday English": general}.items():
        f.write(f"## {name}\\n\\n{df.round(3).to_markdown(index=False)}\\n\\n")
for png in sorted(glob.glob(f"{R}/figures/*.png")):
    display(Image(png))'''),
            ("md", "**Publish the demo** (optional). Creates a public Hugging Face Space under your account with the app, "
                   "these results and figures. Audio is not uploaded: the app fetches clips from the TORGO dataset. "
                   "Needs your `HF_TOKEN` secret and both models uploaded from notebook 3."),
            ("code", '''DEPLOY_SPACE = False  # True publishes the demo as a public Hugging Face Space
if DEPLOY_SPACE:
    import shutil, tempfile
    from huggingface_hub import HfApi, whoami
    user = whoami()["name"]
    space = f"{user}/clear-voice"
    stage = os.path.join(tempfile.mkdtemp(), "space")
    shutil.copytree("app", stage, ignore=shutil.ignore_patterns("__pycache__", "flagged"))
    shutil.copytree("clearvoice", f"{stage}/clearvoice", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(f"{R}/demo_clips.csv", stage)
    shutil.copy(f"{R}/summary.md", stage)
    shutil.copytree(f"{R}/figures", f"{stage}/figures")
    api = HfApi()
    api.create_repo(space, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.add_space_variable(space, "CV_MODEL_A", f"{user}/clear-voice-holda")
    api.upload_folder(folder_path=stage, repo_id=space, repo_type="space")
    print("Demo: https://huggingface.co/spaces/" + space)'''),
        ],
    ),
}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, book in BOOKS.items():
        nbformat.write(book, OUT / f"{name}.ipynb")
        print("wrote", name)

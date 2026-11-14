# Tonight: running Clear Voice on Colab Pro

All code is written and tested on the Mac. What's left needs your accounts and a GPU.
Times below are estimates; check the real ones as you go.

## 1. Setup (about 15 minutes, only you can do these)

1. **GitHub repo.** It must be public so Colab can clone it and judges can read it; the repo holds code, split indices and results only, no audio. Tell Claude "create the repo": it merges the finished build into `main` first, then publishes it. (Pushing before that merge would leave `main` empty, and the notebooks would fail to install.)

   The notebooks expect `https://github.com/armaan-khoja/clear-voice`. If you use another name, change `REPO_URL` in `scripts/build_notebooks.py` and rerun `python scripts/build_notebooks.py`.
2. **Colab Pro.** Subscribe at colab.research.google.com/signup. Keep an eye on the compute-units counter.
3. **Hugging Face.**
   - Create a free account at huggingface.co.
   - Go to Settings > Access Tokens and create a token with **write** access.
   - In Colab, click the key icon (Secrets), add `HF_TOKEN` with that token, and switch on notebook access.
4. **Google Drive.** Make sure about 5 GB is free; models and results go to `MyDrive/clearvoice`.

## 2. Run order

Open each notebook in Colab: File > Open notebook > GitHub, enter `armaan-khoja/clear-voice`, then pick the notebook. Each time, choose Runtime > Change runtime type > **L4 GPU**, then run the setup cell first. It will ask to connect Google Drive; allow it.

| Step | Notebook | Setting | Time (estimate) |
|---|---|---|---|
| 0 | `03_finetune` | `SMOKE = True` | 5–10 min. A full dress rehearsal on a few clips; it writes to `MyDrive/clearvoice/smoke` and never uploads, so it can't overwrite real results |
| 1 | `01_data_and_splits` | | 3 min, GPU not needed |
| 2 | `02_zero_shot_baselines` | | 30–45 min |
| 3 | `03_finetune` | `HELDOUT = "A"` (`PUSH_TO_HUB` is already `True`) | 60–90 min |
| 4 | `03_finetune` | `HELDOUT = "B"` | 45–75 min |
| 5 | `04_personalise` (Sunday) | | about 30 min |
| 6 | `05_results_and_failures` (Sunday) | `DEPLOY_SPACE = True` once results look right | 5 min |

- **Keep the Colab tab open overnight.** Colab Pro does not keep running when the tab closes.
- **If a run disconnects:** rerun the notebook from the top.
  - If it dropped during training, training starts again from the beginning.
  - If training had already finished, the model is on Drive and the notebook skips straight to testing and uploading. Set `RETRAIN = True` only if you want to train again.
- **Training writes to Colab's own disk,** and only the final model (about 1 GB per fold) is copied to Drive.
- **SMOKE is a setting at the top of the setup cell.** Set it back to `False` for the real runs.

## 3. Getting results back into the repo

- Results land in Drive at `MyDrive/clearvoice/results/`. Either download that folder into this repo's `results/` folder, or ask Claude to fetch it with the Google Drive connector.
- Claude then fills in the `TODO(results)` parts of `README.md`.
- To save notebooks with their outputs (judges want outputs visible), use File > Save a copy in GitHub in each notebook, or download the `.ipynb` into `notebooks/`.

## 4. Sunday checklist

- [ ] Send `docs/therapist_email.md` to a speech therapist (today if possible), and quote any reply in the write-up.
- [ ] README results filled in, team names added, and the NP-TORGO citation title checked.
- [ ] Listen to the 10 worst clips (notebook 5) and write one line per clip on why it failed.
- [ ] 2-minute video: about 15 s problem, 60 s demo, 30 s how it works and baseline comparison, 15 s limitation and next step.
- [ ] **First submission by 8 PM ET Sunday; final by 10 PM ET.** Before submitting, open every link in a private window.
- [ ] On the form, choose the Languages track; list every dataset, model, tutorial and AI tool (all listed in README > Sources).
- [ ] After the final submission, tag the repo:

  ```bash
  git tag v1-submission && git push --tags
  ```

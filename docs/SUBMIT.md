# Submitting Clear Voice (Sunday)

First submission by 8 PM ET, final by 10 PM ET. Submissions close at 11:59 PM ET.

## Ready now

- **Repo:** github.com/armaan-khoja/clear-voice. The README has the final results with fine-tuned whisper-medium, every model we tried, the leakage test, personalisation, and where it fails.
- **Three versions of the demo video, about 2 minutes 13 seconds each,** in `video/out/`:
  - `ClearVoice-Exam.mp4`: styled as an exam paper.
  - `ClearVoice-Lab.mp4`: styled as a lab analysis.
  - `ClearVoice-Proof.mp4`: a claim, then the evidence.

  They have music and sound effects but no narration yet.
- **Narration script:** `video/SCRIPT.md`, timed to the video, with the real numbers.

## Only you can do these

1. **Keep the Mac plugged in** and the Colab tab open until about 4:30 PM. The battery died overnight and stopped the last run.
2. **Team names.** Tell Claude, and they go into the README and the closing shot.
3. **Pick a look:** Exam, Lab or Proof.
4. **Record the narration.** Read `video/SCRIPT.md` on a phone in a quiet room. Save it as `video/public/voiceover.wav`, then tell Claude to re-render. The music drops under your voice automatically.
5. **Optional, for the best opening clip:** when macOS asks whether Claude can access your Downloads folder, allow it. Claude can then read the full per-clip predictions and pick the opening example by the documented rule. Without it, the opening comes from the random sample by a second documented rule.
6. **Optional, the live demo:** the Gradio app in `app/` is built for a Hugging Face Space. Publishing it needs your Hugging Face token in Colab's Secrets (`HF_TOKEN`). Claude never handles the token.

## Running now

Fine-tuning whisper-large-v3-turbo on both groups of speakers, on Colab's own disk, finishing around 4:20 PM. If it beats whisper-medium on validation, it becomes Clear Voice: the tables, README and videos are rebuilt from it. If not, it is listed as a candidate.

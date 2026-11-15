# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **Primary audience for the demo video and write-up:** the Veritas AI International High School Hackathon judges, scoring on technical work, originality, impact, testing and honesty, and communication. They are technical and skeptical; they want to see what was built and whether the test is fair.
- **The people the product is for:** people with dysarthria (speech affected by cerebral palsy, ALS, stroke or Parkinson's) who are misheard by dictation and live captions, and the speech therapists who support them.

## Product Purpose

Clear Voice is a speech recogniser fine-tuned for dysarthric speech, plus an honest test of it: it is scored on speakers and sentences it never heard in training, and it reports how much easier tests would have flattered it. Success is fewer words lost for unseen dysarthric speakers than stock Whisper, shown with intervals, without leakage.

## Positioning

Most fine-tuning demos on TORGO score a random split, where the model has already heard the test sentences. Clear Voice reports the strict split (new speaker, new sentence) as its headline and shows the inflation side by side. It also measures how many minutes of a person's own voice personalisation needs.

## Operating Context

- Built in 36 hours by a high-school team; training on Google Colab (L4 GPU); code at github.com/armaan-khoja/clear-voice.
- Deliverables: the repository and README write-up, a Gradio demo on Hugging Face Spaces, and a demo video of about 2 minutes narrated by the team.
- The same project may be entered in the Conrad Challenge, where all elements must be the team's own work and AI help must be cited.

## Capabilities and Constraints

- whisper-medium fine-tuned on TORGO (Hugging Face mirror abnerh/TORGO-database), two speaker folds, chosen on validation over fine-tuned whisper-small and whisper-large-v3-turbo; baselines are stock whisper-medium (its base), whisper-small and whisper-large-v3.
- Metrics: word error rate on sentences, character error rate on single words, speaker-level bootstrap 95% intervals, per-clip medians.
- Research prototype, not a medical device. TORGO is non-commercial research data: audio may appear in the demo and video, never in the repository.
- Only 8 dysarthric speakers exist in TORGO; this is a stated limitation.

## Brand Commitments

- Name: Clear Voice. No logo, palette or fonts were given; the team delegated the identity.
- Voice: plain, specific, honest about limits. Numbers come only from the project's own results.

## Evidence on Hand

- Results on Google Drive (MyDrive/clearvoice/results): predictions per system and fold, leakage table, LibriSpeech check, personalisation curve, demo clips chosen at random per severity.
- Real recordings of unseen speakers (TORGO) for demo moments.
- No testimonials or user studies yet. A speech therapist has been emailed; no reply is on hand. Do not invent quotes, users or adoption.

## Product Principles

1. The hard number is the headline. Never show a flattering split as the result.
2. Every claim on screen comes from the project's own results files.
3. The speaker's real voice is the centre of the demo, heard clearly.
4. Name limits plainly (8 dysarthric speakers, read prompts, not a medical device).

## Accessibility & Inclusion

The video must be understandable with sound off (on-screen transcripts carry the demo) and readable from a distance (large type, high contrast). Avoid portraying people with dysarthria as objects of pity.

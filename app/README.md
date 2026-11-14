---
title: Clear Voice
emoji: 🗣️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
short_description: Speech recognition for people with dysarthria, tested honestly
---

# Clear Voice

Whisper fine-tuned so people with dysarthria are understood, tested on speakers and sentences it never heard.

Research prototype, not a medical device. Uses the TORGO database (Rudzicz, Namasivayam and Wolff, 2012),
which is free for non-commercial research use. Audio is fetched from the TORGO dataset on Hugging Face at
runtime and is not stored in this Space.

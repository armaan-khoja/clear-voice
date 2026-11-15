# Clear Voice: narration script

About 96 seconds. The same script fits all three looks (Exam, Lab, Proof).

How to record: read it once at a calm pace on a phone in a quiet room, about 15 cm from your mouth. Leave silence wherever a recording plays, so the speaker is heard on their own. Save it as `video/public/voiceover.wav` and tell Claude; the music then drops under your voice.

| Starts | Scene | Say |
|---|---|---|
| 0:00.0 | title | Three speech models are about to be tested on voices they've never heard. |
| 0:05.1 | hook | [Let the recording play. Start speaking once it ends.] Stock Whisper heard this. Our model, Clear Voice, heard this. And this is what was actually said. |
| 0:18.0 | problem | That speaker has dysarthria, which can come with cerebral palsy, ALS, a stroke or Parkinson's. Stock Whisper gets 5 percent of typical speakers' words wrong, and 60 percent of theirs. |
| 0:30.6 | demo | Three more speakers, picked at random. First, mild. [Then silence while the 3-second recording plays.] |
| 0:42.2 | how | We fine-tuned OpenAI's Whisper on more than 16,000 recordings from the TORGO database. What matters most is the test: the people and sentences we test on never appear in training. |
| 0:55.2 | results | Here's the result on new speakers saying new sentences. Clear Voice gets 30 percent of words wrong. Stock Whisper gets 60 percent, and a model six times bigger gets 45 percent. |
| 1:06.8 | honest | Tested on sentences it had practised, its error rate would be 12 percent: 18 points better, and not real. So we report the harder number. |
| 1:17.4 | personal | 10 minutes of one person's own voice helps a lot: for M01, with severe dysarthria, errors fell from 60 to 30 percent. |
| 1:27.0 | close | Next, we want to try it with speech therapists on real dictation. We're Team names here, and this is Clear Voice. |

Numbers come from `video/src/data.json`, which `scripts/video_data.py` builds from the results on Google Drive. Rerun this script whenever the results change.

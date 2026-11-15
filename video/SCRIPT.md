# Clear Voice: narration script

About 133 seconds. The same script fits all three looks (Exam, Lab, Proof).

How to record: read it once at a calm pace on a phone in a quiet room, about 15 cm from your mouth. Leave silence wherever a recording plays, so the speaker is heard on their own. Save it as `video/public/voiceover.wav` and tell Claude; the music then drops under your voice.

| Starts | Scene | Say |
|---|---|---|
| 0:00.0 | title | Three speech models are about to be tested on voices they've never heard. |
| 0:05.1 | hook | [Let the recording play. Start speaking once it ends.] Stock Whisper heard this. Our model, Clear Voice, heard this. And this is what was actually said. |
| 0:18.0 | problem | That speaker has dysarthria, which can come with cerebral palsy, ALS, a stroke or Parkinson's. Stock Whisper gets 4 percent of typical speakers' words wrong, and 34 percent of theirs. |
| 0:30.6 | demo | Three more speakers, picked at random. First, mild. [Then silence while the 3-second recording plays.] All three get this one right. |
| 0:42.9 | demo | Moderate. [Then silence while the 3-second recording plays.] Here they make the same number of mistakes. |
| 0:54.6 | demo | And severe. [Then silence while the 7-second recording plays.] Here stock Whisper does better. Picked at random means it doesn't always win. |
| 1:10.9 | how | We fine-tuned OpenAI's Whisper on more than 16,000 recordings from the TORGO database. What matters most is the test: the people and sentences we test on never appear in training. |
| 1:23.8 | results | Here's the result on new speakers saying new sentences. Clear Voice gets 29 percent of words wrong. The model it started from gets 34, and a model twice its size gets 22. On single words, Clear Voice is the best of the three: 39 percent of letters wrong, against 96 for the model it started from. |
| 1:43.4 | honest | Tested on sentences it had practised, its error rate would be 16 percent: 14 points better, and not real. So we report the harder number. |
| 1:54.0 | personal | Ten minutes of one person's own voice helped a little: for M01, with severe dysarthria, errors fell from 58 to 52 percent. |
| 2:03.6 | close | Next, we want to try it with speech therapists on real dictation. We're Team names here, and this is Clear Voice. |

Numbers come from `video/src/data.json`, which `scripts/video_data.py` builds from the results on Google Drive. Rerun this script whenever the results change.

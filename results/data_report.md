# TORGO data report

Clips: 16552

## Per speaker (minutes count both microphone copies)

| speaker   |   clips |   minutes | severity        | status     | gender   |
|:----------|--------:|----------:|:----------------|:-----------|:---------|
| F01       |     236 |       9.6 | severe          | dysarthria | female   |
| F03       |    1087 |      49.8 | moderate        | dysarthria | female   |
| F04       |     673 |      36.7 | mild            | dysarthria | female   |
| FC01      |     302 |      18   | typical         | healthy    | female   |
| FC02      |    2187 |     103.7 | typical         | healthy    | female   |
| FC03      |    1922 |      83.3 | typical         | healthy    | female   |
| M01       |     743 |      48.9 | severe          | dysarthria | male     |
| M02       |     772 |      47.3 | severe          | dysarthria | male     |
| M03       |     810 |      42.4 | mild            | dysarthria | male     |
| M04       |     666 |      47.9 | severe          | dysarthria | male     |
| M05       |     587 |      45.1 | moderate-severe | dysarthria | male     |
| MC01      |    2155 |     108.5 | typical         | healthy    | male     |
| MC02      |    1123 |      66.2 | typical         | healthy    | male     |
| MC03      |    1669 |      58.9 | typical         | healthy    | male     |
| MC04      |    1620 |      54.2 | typical         | healthy    | male     |

## Microphones

{'arrayMic': 8697, 'headMic': 7855}

Unique utterances (speaker, session, utt): 9075

Unique normalised prompts: 959 (words: 616, sentences: 343)

Empty normalised transcripts: 0

Utterances whose mic copies disagree on transcript: 0

Exact duplicate filenames: 0

Decoded 50 random clips: length / (duration x 16 kHz) ranges 1.000 to 1.000

## Status values

{'healthy': 10978, 'dysarthria': 5574}

## Severity values

{'typical': 10978, 'moderate-severe': 587, 'severe': 2417, 'mild': 1483, 'moderate': 1087}

## Splits (seed 42)

Prompts: train 670, val 96, test 193

Near-duplicate prompt groups kept in one split (similarity > 0.8): 90, e.g. [['2 other cases also were under advisement', '2 other cases were also under advisement'], ['carl lives in a lively home', 'carl lives in a lovely home'], ['except in the winter when the ooze or snow or ice prevents', 'except in the winter when the ozone or snow or ice prevents']]

Validation or test prompts more than 0.8 similar to a training prompt: 0

### Model that never hears fold A (F01, M01, F03, M03, FC01, MC01, MC02)

| part         |   clips |   speakers |   dysarthric clips |   minutes |
|:-------------|--------:|-----------:|-------------------:|----------:|
| train        |    6766 |          8 |               1747 |     319.4 |
| val          |     650 |          8 |                222 |      34.1 |
| test_seen    |    2227 |          7 |                946 |     111.4 |
| test_strict  |     737 |          7 |                318 |      35.9 |
| test_newsent |    1257 |          8 |                372 |      57.9 |

Strict test clips per held-out speaker (fold A):

| speaker   |   clips |   words |   sentences |
|:----------|--------:|--------:|------------:|
| F01       |      25 |      23 |           2 |
| F03       |     122 |      94 |          28 |
| FC01      |      35 |      29 |           6 |
| M01       |      81 |      64 |          17 |
| M03       |      90 |      72 |          18 |
| MC01      |     235 |     182 |          53 |
| MC02      |     149 |     116 |          33 |

### Model that never hears fold B (M02, M04, M05, F04, FC02, FC03, MC03, MC04)

| part         |   clips |   speakers |   dysarthric clips |   minutes |
|:-------------|--------:|-----------:|-------------------:|----------:|
| train        |    4284 |          7 |               1884 |     223.2 |
| val          |     384 |          7 |                178 |      20.5 |
| test_seen    |    3820 |          8 |               1085 |     182.4 |
| test_strict  |    1257 |          8 |                372 |      57.9 |
| test_newsent |     737 |          7 |                318 |      35.9 |

Strict test clips per held-out speaker (fold B):

| speaker   |   clips |   words |   sentences |
|:----------|--------:|--------:|------------:|
| F04       |      90 |      71 |          19 |
| FC02      |     264 |     204 |          60 |
| FC03      |     207 |     161 |          46 |
| M02       |      87 |      67 |          20 |
| M04       |      95 |      72 |          23 |
| M05       |     100 |      78 |          22 |
| MC03      |     197 |     157 |          40 |
| MC04      |     217 |     167 |          50 |

### Personalisation pools (one copy per utterance, training prompts only)

| speaker   | 2 min    | 5 min     | 10 min    | 20 min   |
|:----------|:---------|:----------|:----------|:---------|
| M01       | 33 clips | 85 clips  | 182 clips | capped   |
| F03       | 46 clips | 112 clips | 240 clips | capped   |
| M03       | 41 clips | 105 clips | 207 clips | capped   |
| M04       | 27 clips | 64 clips  | 127 clips | capped   |

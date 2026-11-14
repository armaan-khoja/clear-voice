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

Prompts: train 671, val 96, test 192

### Model that never hears fold A (F01, M01, F03, M03, FC01, MC01, MC02)

| part         |   clips |   speakers |   dysarthric clips |   minutes |
|:-------------|--------:|-----------:|-------------------:|----------:|
| train        |    7149 |          8 |               1939 |     337   |
| val          |     509 |          8 |                137 |      23.4 |
| test_seen    |    2387 |          7 |               1031 |     119.1 |
| test_strict  |     673 |          7 |                293 |      34.6 |
| test_newsent |    1158 |          8 |                336 |      56.6 |

Strict test clips per held-out speaker (fold A):

| speaker   |   clips |   words |   sentences |
|:----------|--------:|--------:|------------:|
| F01       |      19 |      17 |           2 |
| F03       |     114 |      85 |          29 |
| FC01      |      29 |      25 |           4 |
| M01       |      77 |      58 |          19 |
| M03       |      83 |      61 |          22 |
| MC01      |     215 |     157 |          58 |
| MC02      |     136 |     102 |          34 |

### Model that never hears fold B (M02, M04, M05, F04, FC02, FC03, MC03, MC04)

| part         |   clips |   speakers |   dysarthric clips |   minutes |
|:-------------|--------:|-----------:|-------------------:|----------:|
| train        |    4599 |          7 |               2055 |     239.4 |
| val          |     288 |          7 |                118 |      14.1 |
| test_seen    |    4060 |          8 |               1206 |     194.5 |
| test_strict  |    1158 |          8 |                336 |      56.6 |
| test_newsent |     673 |          7 |                293 |      34.6 |

Strict test clips per held-out speaker (fold B):

| speaker   |   clips |   words |   sentences |
|:----------|--------:|--------:|------------:|
| F04       |      87 |      65 |          22 |
| FC02      |     246 |     185 |          61 |
| FC03      |     195 |     141 |          54 |
| M02       |      82 |      62 |          20 |
| M04       |      72 |      58 |          14 |
| M05       |      95 |      69 |          26 |
| MC03      |     178 |     132 |          46 |
| MC04      |     203 |     145 |          58 |

### Personalisation pools (one copy per utterance, training prompts only)

| speaker   | 2 min    | 5 min     | 10 min    | 20 min    |
|:----------|:---------|:----------|:----------|:----------|
| M01       | 37 clips | 89 clips  | 174 clips | capped    |
| F03       | 50 clips | 128 clips | 251 clips | capped    |
| M03       | 43 clips | 105 clips | 207 clips | capped    |
| M04       | 29 clips | 76 clips  | 145 clips | 264 clips |

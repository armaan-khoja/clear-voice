"""Write video/SCRIPT.md: the narration for the demo video, timed to the shared timeline, with real numbers.

Run after scripts/video_data.py: .venv/bin/python scripts/video_script.py
The same script fits all three looks (Exam, Lab, Proof), because they share video/src/shared/timeline.json.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = json.loads((ROOT / "video/src/shared/timeline.json").read_text())
D = json.loads((ROOT / "video/src/data.json").read_text())


def pct(x: float) -> str:
    return f"{round(x * 100)} percent"


def stamp(frame: int) -> str:
    s = frame / T["fps"]
    return f"{int(s // 60)}:{s % 60:04.1f}"


def main() -> None:
    demos = D["demo"] or [D["opening"]]
    order = ["title", "hook", "problem"] + ["demo"] * len(demos) + ["how", "results", "honest", "personal", "close"]
    starts, t = [], 0
    for i, kind in enumerate(order):
        starts.append(t)
        t += T["frames"][kind] - T["overlap"]
    total = t + T["overlap"]

    h = {b["name"]: b for b in D["headline"]}
    stock, large, ours = h["whisper-small"]["value"], h["whisper-large-v3"]["value"], h["Clear Voice"]["value"]
    seen = next(b["value"] for b in D["leakage"] if "seen sentence" in b["name"].lower())
    strict = next(b["value"] for b in D["leakage"] if "new speaker, new" in b["name"].lower())
    p = D["personalisation"][0] if D["personalisation"] else None
    typical = D.get("stockTypical")
    clip_secs = [D["opening"]["duration"]] + [c["duration"] for c in demos]

    if ours < min(stock, large):
        verdict = f"Clear Voice gets {pct(ours)} of words wrong. Stock Whisper gets {pct(stock)}, and a model six times bigger gets {pct(large)}."
    elif ours < stock:
        verdict = f"Clear Voice gets {pct(ours)} of words wrong, against {pct(stock)} for the model it started from. The model six times bigger still does better, at {pct(large)}."
    else:
        verdict = f"Clear Voice gets {pct(ours)} of words wrong. Stock Whisper gets {pct(stock)}, so on this test it is not better yet."

    lines = {
        "title": "Three speech models are about to be tested on voices they've never heard.",
        "hook": "[Let the recording play. Start speaking once it ends.] Stock Whisper heard this. Our model, Clear Voice, heard this. And this is what was actually said.",
        "problem": "That speaker has dysarthria, which can come with cerebral palsy, ALS, a stroke or Parkinson's. "
        + (f"Stock Whisper gets {pct(typical)} of typical speakers' words wrong, and {pct(stock)} of theirs." if typical is not None else f"Stock Whisper gets {pct(stock)} of these speakers' words wrong."),
        "how": f"We fine-tuned OpenAI's Whisper on more than {D['nRecordings'] // 1000 * 1000:,} recordings from the TORGO database. What matters most is the test: the people and sentences we test on never appear in training.",
        "results": f"Here's the result on new speakers saying new sentences. {verdict}",
        "honest": (f"Tested on sentences it had practised, its error rate would be {pct(seen)}: {round((strict - seen) * 100)} points better, and not real. So we report the harder number."
                   if strict > seen else f"Practised sentences gave it no advantage here: {pct(seen)} against {pct(strict)}. We report the harder test anyway."),
        "personal": (f"{({2: 'Two', 5: 'Five', 10: 'Ten'}).get(p['points'][-1]['minutes'], str(p['points'][-1]['minutes']))} minutes of one person's own voice helps a lot: for {p['speaker']}, with {p['severity']} dysarthria, errors fell from {round(p['points'][0]['wer'] * 100)} to {pct(p['points'][-1]['wer'])}." if p else ""),
        "close": f"Next, we want to try it with speech therapists on real dictation. We're {D['team']}, and this is Clear Voice.",
    }
    demo_lines = ["Three more speakers, picked at random. First, mild.", "Moderate.", "And severe."]

    out = [
        "# Clear Voice: narration script",
        "",
        f"About {round(total / T['fps'])} seconds. The same script fits all three looks (Exam, Lab, Proof).",
        "",
        "How to record: read it once at a calm pace on a phone in a quiet room, about 15 cm from your mouth. Leave silence wherever a recording plays, so the speaker is heard on their own. Save it as `video/public/voiceover.wav` and tell Claude; the music then drops under your voice.",
        "",
        "| Starts | Scene | Say |",
        "|---|---|---|",
    ]
    d = 0
    for kind, start in zip(order, starts):
        if kind == "demo":
            say = f"{demo_lines[min(d, 2)]} [Then silence while the {clip_secs[1 + d]:.0f}-second recording plays.]"
            d += 1
        else:
            say = lines[kind]
        out.append(f"| {stamp(start)} | {kind} | {say} |")
    out += ["", "Numbers come from `video/src/data.json`, which `scripts/video_data.py` builds from the results on Google Drive. Rerun this script whenever the results change.", ""]
    (ROOT / "video/SCRIPT.md").write_text("\n".join(out))
    print("wrote video/SCRIPT.md")


if __name__ == "__main__":
    main()

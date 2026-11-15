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


def personal_line(p: dict) -> str:
    first, last = p["points"][0]["wer"], p["points"][-1]["wer"]
    mins = {2: "Two", 5: "Five", 10: "Ten"}.get(p["points"][-1]["minutes"], str(p["points"][-1]["minutes"]))
    drop = round((first - last) * 100)
    how = "helped a lot" if drop >= 10 else "helped a little" if drop >= 3 else "barely helped"
    change = f"errors fell from {round(first * 100)} to {pct(last)}" if drop > 0 else f"errors stayed at {pct(last)}"
    return f"{mins} minutes of one person's own voice {how}: for {p['speaker']}, with {p['severity']} dysarthria, {change}."


def demo_remark(clip: dict, base: str) -> str:
    """One honest sentence after a random demo recording, from the two transcripts' error rates."""
    ours, stock = clip["systems"].get("Clear Voice", {}).get("wer"), clip["systems"].get(base, {}).get("wer")
    if ours is None or stock is None:
        return ""
    if ours < stock:
        return " Clear Voice gets closer."
    if ours > stock:
        return " Here stock Whisper does better. Picked at random means it doesn't always win."
    return " Here they make the same number of mistakes." if ours > 0 else " All three get this one right."


def stamp(frame: int) -> str:
    s = frame / T["fps"]
    return f"{int(s // 60)}:{s % 60:04.1f}"


def main() -> None:
    demos = D["demo"] or [D["opening"]]
    order = ["title", "hook", "problem"] + ["demo"] * len(demos) + ["how", "results", "honest", "personal", "close"]
    starts, t = [], 0
    clips = iter(demos)
    for i, kind in enumerate(order):
        starts.append(t)
        frames = T["frames"][kind]
        if kind == "demo":  # same rule as demoFrames in video/src/shared/timeline.ts
            frames = max(frames, T["demoPlay"] + round(next(clips)["duration"] * T["fps"]) + T["afterClip"])
        t += frames - T["overlap"]
    total = t + T["overlap"]

    base = D.get("stockName", "whisper-small")
    bigger = {2: "twice its size", 3: "three times its size", 6: "six times bigger"}.get(D.get("largeRatio", 6), f"{D.get('largeRatio')} times bigger")
    h = {b["name"]: b for b in D["headline"]}
    w = {b["name"]: b["value"] for b in D.get("headlineWords", [])}
    stock, large, ours = h[base]["value"], h["whisper-large-v3"]["value"], h["Clear Voice"]["value"]
    seen = next(b["value"] for b in D["leakage"] if "seen sentence" in b["name"].lower())
    strict = next(b["value"] for b in D["leakage"] if "new speaker, new" in b["name"].lower())
    p = D["personalisation"][0] if D["personalisation"] else None
    typical = D.get("stockTypical")
    clip_secs = [D["opening"]["duration"]] + [c["duration"] for c in demos]

    if ours < min(stock, large):
        verdict = f"Clear Voice gets {pct(ours)} of words wrong. The model it started from gets {pct(stock)}, and a model {bigger} gets {pct(large)}."
    elif ours < stock:
        verdict = f"Clear Voice gets {pct(ours)} of words wrong, the model it started from {round(stock * 100)}, and a model {bigger} {round(large * 100)}. With only eight test speakers, the first two ranges overlap."
    else:
        verdict = f"Clear Voice gets {pct(ours)} of words wrong. Stock Whisper gets {pct(stock)}, so on this test it is not better yet."
    if "Clear Voice" in w and base in w and w["Clear Voice"] < min(w[base], w.get("whisper-large-v3", 9)):
        verdict += f" On single words, Clear Voice is the best of the three: {pct(w['Clear Voice'])} of letters wrong, against {round(w[base] * 100)} for the model it started from."

    lines = {
        "title": "Three speech models are about to be tested on voices they've never heard.",
        "hook": "[Let the recording play. Start speaking once it ends.] Stock Whisper heard this. Our model, Clear Voice, heard this. And this is what was actually said.",
        "problem": "That speaker has dysarthria, which can come with cerebral palsy, ALS, a stroke or Parkinson's. "
        + (f"Stock Whisper gets {pct(typical)} of typical speakers' words wrong, and {pct(stock)} of theirs." if typical is not None else f"Stock Whisper gets {pct(stock)} of these speakers' words wrong."),
        "how": f"We fine-tuned OpenAI's Whisper on more than {D['nRecordings'] // 1000 * 1000:,} recordings from the TORGO database. What matters most is the test: the people and sentences we test on never appear in training.",
        "results": f"Here's the result on new speakers saying new sentences. {verdict}",
        "honest": (f"Tested on sentences it had practised, its error rate would be {pct(seen)}: {round((strict - seen) * 100)} points better, and not real. So we report the harder number."
                   if strict > seen else f"Practised sentences gave it no advantage here: {pct(seen)} against {pct(strict)}. We report the harder test anyway."),
        "personal": (personal_line(p) if p else ""),
        "close": f"Next, we want to try it with speech therapists on real dictation. We're {D['team']}, and this is Clear Voice.",
    }
    count = {1: "One more speaker", 2: "Two more speakers", 3: "Three more speakers"}.get(len(demos), f"{len(demos)} more speakers")
    demo_lines = [f"{count}, picked at random. First, {demos[0]['severity']}."] + [
        (f"And {c['severity'].replace('-', ' ')}." if i == len(demos) - 1 else c["severity"].replace("-", " ").capitalize() + ".") for i, c in enumerate(demos[1:], 1)]

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
            say = f"{demo_lines[d]} [Then silence while the {clip_secs[1 + d]:.0f}-second recording plays.]{demo_remark(demos[d], base)}"
            d += 1
        else:
            say = lines[kind]
        out.append(f"| {stamp(start)} | {kind} | {say} |")
    out += ["", "Numbers come from `video/src/data.json`, which `scripts/video_data.py` builds from the results on Google Drive. Rerun this script whenever the results change.", ""]
    (ROOT / "video/SCRIPT.md").write_text("\n".join(out))
    print("wrote video/SCRIPT.md")


if __name__ == "__main__":
    main()

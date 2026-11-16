"""Test Say It's sentence corrector (a teammate's browser app) on Clear Voice's strict-split outputs.

Say It turns a recogniser's text into candidate sentences (similar-sounding words, dropped short words put back,
everyday phrases) and a set of gradient-boosted decision trees picks one. It works on text, so it can run on any
model's output. This script takes the corrector out of the app's HTML, runs it in Node exactly as the app's
finish() does (repeat removal, then famPick), and scores it with the same code as the headline table.

The decision to use it is made on validation only (mean validation WER over both folds, as for the model choice).

Run: .venv/bin/python scripts/sayit/evaluate.py <say-it.html> <results folder> <output folder>
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clearvoice import report  # noqa: E402
from clearvoice.metrics import score  # noqa: E402
from clearvoice.pipeline import read_preds  # noqa: E402

HERE = Path(__file__).resolve().parent


def extract_corrector(html_path: str, out: Path) -> Path:
    """The corrector's code and data from the app, as a Node module (no audio is copied)."""
    t = open(html_path, encoding="utf-8").read()
    scripts = re.findall(r"<script>(.*?)</script>", t, re.S)
    data = next(s for s in scripts if s.lstrip().startswith("var FAMDATA="))
    code = next(s for s in scripts if "FAMILIAR PATTERNS AND THE SENTENCE TREE" in s)
    mod = re.search(r'<script type="module">(.*?)</script>', t, re.S).group(1)
    norm = re.search(r"function norm\(s\)\{.*?\n", mod).group(0)
    norep = re.search(r"function noRepeats\(s\)\{.*?\n  return w\.join\(' '\);\n\}", mod, re.S).group(0)
    js = out / "sayit_corrector.js"
    js.write_text(data + "\n" + code.replace("if(typeof module!=='undefined') module.exports=", "var __exp=") + "\n"
                  + norm + norep + "\nfamLoad(FAMDATA);\nmodule.exports={famPick:famPick,noRepeats:noRepeats,FAM:FAM};\n")
    return js


def correct(rows: list, out: Path, name: str) -> dict:
    src, dst = out / f"{name}_in.jsonl", out / f"{name}_out.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows))
    subprocess.run(["node", str(out / "run.js"), str(src), str(dst)], check=True)  # the copy beside the corrector
    return {str(x["id"]): x for x in map(json.loads, dst.read_text().splitlines())}


def main(html: str, results: str, out_dir: str) -> None:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.js").write_text((HERE / "run.js").read_text())
    extract_corrector(html, out)
    R = Path(results)

    val = pd.concat([read_preds(R / f"val_ft_hold{f}_medium.csv") for f in "AB"], ignore_index=True)
    key = val["fold"] + val["row_id"].astype(str)
    vo = correct([{"id": k, "a": h, "b": None} for k, h in zip(key, val["hyp"])], out, "val")
    rows = []
    for name, col in [("Clear Voice", None), ("Clear Voice + repeat removal", "a_norep"), ("Clear Voice + Say It corrector", "a_fix")]:
        h = val["hyp"] if col is None else key.map(lambda k: vo[k][col])
        per = {f: score(list(val["ref"][val["fold"] == f]), list(h[val["fold"] == f]))["wer"] for f in "AB"}
        rows.append({"system": name, "val_wer_A": per["A"], "val_wer_B": per["B"], "val_wer_mean": (per["A"] + per["B"]) / 2})
    pd.DataFrame(rows).to_csv(out / "sayit_validation.csv", index=False)

    cv = pd.concat([read_preds(R / f"preds_ft_hold{f}_medium.csv") for f in "AB"], ignore_index=True)
    cv = cv[cv["part"] == "test_strict"].reset_index(drop=True)
    lv3 = read_preds(R / "preds_zeroshot_whisper-large-v3.csv")
    lv3 = lv3[lv3["part"] == "test_strict"].set_index("row_id").loc[cv["row_id"]].reset_index()
    to = correct([{"id": int(i), "a": a, "b": b} for i, a, b in zip(cv["row_id"], cv["hyp"], lv3["hyp"])], out, "test")
    ids = cv["row_id"].astype(str)

    def using(df, col):
        return df.assign(hyp=ids.map(lambda k: to[k][col]).values)

    systems = {"whisper-large-v3 (stock)": lv3, "whisper-large-v3 + Say It corrector": using(lv3, "b_fix"),
               "Clear Voice": cv, "Clear Voice + repeat removal": using(cv, "a_norep"),
               "Clear Voice + Say It corrector": using(cv, "a_fix"),
               "Clear Voice + large-v3 + Say It corrector": using(cv, "both")}
    report.headline(systems).to_csv(out / "sayit_headline.csv", index=False)

    d = cv[cv["is_dysarthric"].astype(bool)]
    changed = [(r.ref, to[str(r.row_id)]["a_norep"], to[str(r.row_id)]["a_fix"]) for r in d.itertuples()
               if to[str(r.row_id)]["a_fix"] != to[str(r.row_id)]["a_norep"]]
    better = sum(score([a], [c])["wer"] < score([a], [b])["wer"] for a, b, c in changed)
    worse = sum(score([a], [c])["wer"] > score([a], [b])["wer"] for a, b, c in changed)
    (out / "sayit_changes.txt").write_text(f"changed {len(changed)} of {len(d)}: {better} better, {worse} worse\n"
                                          + "\n".join(f"{a} | {b} -> {c}" for a, b, c in changed))
    print(open(out / "sayit_validation.csv").read(), open(out / "sayit_headline.csv").read(),
          (out / "sayit_changes.txt").read_text().splitlines()[0], sep="\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])

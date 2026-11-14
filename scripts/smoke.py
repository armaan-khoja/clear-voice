"""End-to-end smoke run on real TORGO with whisper-tiny on CPU: predict, fine-tune 2 steps, predict, report."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clearvoice import report  # noqa: E402
from clearvoice.asr import load_model  # noqa: E402
from clearvoice.data import load_audio  # noqa: E402
from clearvoice.pipeline import load_torgo, predict, predict_fold, rows_for  # noqa: E402
from clearvoice.train import TrainConfig, finetune  # noqa: E402

work = ROOT / ".superpowers" / "smoke"
parquets, meta = load_torgo(work)
splits = ROOT / "splits"

model, processor = load_model("openai/whisper-tiny")
zs = predict_fold(model, processor, parquets, meta, splits, "A", ["test_strict"], "whisper-tiny", limit=6)
print(zs[["speaker", "prompt_type", "ref", "hyp"]].to_string())

train = rows_for(meta, splits, "holdA", "train").head(8)
val = rows_for(meta, splits, "holdA", "val").head(4)
audio = load_audio(parquets, meta, list(train.row_id) + list(val.row_id))
cfg = TrainConfig(base_model="openai/whisper-tiny", max_steps=2, eval_steps=1, save_steps=1, batch_size=4,
                  eval_batch_size=4, num_workers=0)
best = finetune(cfg, train, val, audio, tempfile.mkdtemp())
ft_model, ft_proc = load_model(best)
ft = predict_fold(ft_model, ft_proc, parquets, meta, splits, "A", ["test_seen", "test_newsent", "test_strict"],
                  "Clear Voice", limit=6)
print(report.leakage(ft).to_string())
print(report.headline({"whisper-tiny": zs, "Clear Voice": ft}).to_string())
print("SMOKE OK")

import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration

from clearvoice.train import load_base_model


def _tiny_half_model(path):
    """A random, very small Whisper saved in float16, like openai/whisper-large-v3-turbo is published."""
    cfg = WhisperConfig(d_model=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                        decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64, vocab_size=100,
                        num_mel_bins=80, max_source_positions=1500, max_target_positions=64, pad_token_id=1,
                        bos_token_id=2, eos_token_id=2, decoder_start_token_id=3)
    WhisperForConditionalGeneration(cfg).half().save_pretrained(path)


def test_training_loads_a_float16_checkpoint_as_float32(tmp_path):
    # Transformers 5 loads weights in the dtype they were saved in; training needs float32 master weights,
    # otherwise evaluation feeds float32 features into float16 layers and crashes.
    _tiny_half_model(tmp_path / "half")
    model = load_base_model(str(tmp_path / "half"))
    assert {p.dtype for p in model.parameters()} == {torch.float32}

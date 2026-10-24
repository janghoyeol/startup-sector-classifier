import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import GPTClassifier, GPTConfig  # noqa: E402
from predictor import SectorPredictor  # noqa: E402
from tokenizer import HANGUL_JAMO, JamoTokenizer, SPECIAL_TOKENS  # noqa: E402


def make_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    vocab = SPECIAL_TOKENS + HANGUL_JAMO + list(" .,가나다라")
    tokenizer = JamoTokenizer(vocab)
    vocab_path = tmp_path / "vocab.json"
    tokenizer.save(vocab_path)

    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=32, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    label_map = {"가 업종": 0, "나 업종": 1}
    model = GPTClassifier(config, num_classes=len(label_map))

    ckpt_path = tmp_path / "finetune.pt"
    torch.save({"model_state": model.state_dict(), "config": config, "label_map": label_map}, ckpt_path)
    return ckpt_path, vocab_path


def test_predict_returns_ranked_labels_summing_near_one(tmp_path):
    ckpt_path, vocab_path = make_checkpoint(tmp_path)
    predictor = SectorPredictor(checkpoint_path=ckpt_path, vocab_path=vocab_path)

    preds = predictor.predict("가나다라 가나다라")
    assert len(preds) == 2
    assert {p["label"] for p in preds} == {"가 업종", "나 업종"}
    assert preds[0]["probability"] >= preds[1]["probability"]
    assert abs(sum(p["probability"] for p in preds) - 1.0) < 1e-3


def test_predict_empty_text_returns_empty_list(tmp_path):
    ckpt_path, vocab_path = make_checkpoint(tmp_path)
    predictor = SectorPredictor(checkpoint_path=ckpt_path, vocab_path=vocab_path)
    assert predictor.predict("   ") == []


def test_missing_file_without_hf_env_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("SECTOR_MODEL_HF_REPO", raising=False)
    try:
        SectorPredictor(
            checkpoint_path=tmp_path / "does-not-exist.pt",
            vocab_path=tmp_path / "does-not-exist.json",
        )
        assert False, "FileNotFoundError가 났어야 합니다"
    except FileNotFoundError:
        pass

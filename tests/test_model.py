import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import GPT, GPTClassifier, GPTConfig  # noqa: E402


def make_tiny_config() -> GPTConfig:
    return GPTConfig(vocab_size=50, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)


def make_tiny_model() -> GPT:
    return GPT(make_tiny_config())


def test_forward_shape():
    model = make_tiny_model()
    idx = torch.randint(0, 50, (4, 8))
    logits, loss = model(idx)
    assert logits.shape == (4, 8, 50)
    assert loss is None


def test_loss_computed_when_targets_given():
    model = make_tiny_model()
    idx = torch.randint(0, 50, (4, 8))
    targets = torch.randint(0, 50, (4, 8))
    _, loss = model(idx, targets)
    assert loss is not None
    assert loss.item() > 0


def test_backward_computes_gradients():
    model = make_tiny_model()
    idx = torch.randint(0, 50, (4, 8))
    targets = torch.randint(0, 50, (4, 8))
    before = model.tok_emb.weight.clone()
    _, loss = model(idx, targets)
    loss.backward()
    assert model.tok_emb.weight.grad is not None
    assert torch.count_nonzero(model.tok_emb.weight.grad) > 0
    # optimizer.step()을 아직 안 걸었으니 가중치 값 자체는 그대로여야 함
    assert torch.equal(before, model.tok_emb.weight)


def test_generate_appends_tokens_within_vocab():
    model = make_tiny_model()
    idx = torch.randint(0, 50, (1, 3))
    out = model.generate(idx, max_new_tokens=5, top_k=10)
    assert out.shape == (1, 8)
    assert out.min() >= 0 and out.max() < 50


def test_weight_tying():
    model = make_tiny_model()
    assert model.head.weight is model.tok_emb.weight


def make_tiny_classifier(num_classes: int = 3) -> GPTClassifier:
    return GPTClassifier(make_tiny_config(), num_classes=num_classes)


def test_classifier_forward_shape():
    model = make_tiny_classifier(num_classes=3)
    idx = torch.randint(0, 50, (4, 6))
    lengths = torch.tensor([6, 4, 6, 2])
    logits, loss = model(idx, lengths)
    assert logits.shape == (4, 3)
    assert loss is None


def test_classifier_loss_computed_when_targets_given():
    model = make_tiny_classifier(num_classes=3)
    idx = torch.randint(0, 50, (4, 6))
    lengths = torch.full((4,), 6)
    targets = torch.randint(0, 3, (4,))
    _, loss = model(idx, lengths, targets)
    assert loss is not None and loss.item() > 0


def test_classifier_padding_after_last_real_token_does_not_change_pooled_output():
    # causal attention이면 뒤에 패딩을 더 붙여도 "마지막 실제 토큰" 위치의 표현은
    # 그대로여야 한다 (미래 토큰을 못 보니까).
    model = make_tiny_classifier(num_classes=3)
    model.eval()
    idx_short = torch.randint(1, 50, (1, 4))
    idx_padded = torch.cat([idx_short, torch.zeros(1, 2, dtype=torch.long)], dim=1)
    lengths = torch.tensor([4])

    logits_short, _ = model(idx_short, lengths)
    logits_padded, _ = model(idx_padded, lengths)
    assert torch.allclose(logits_short, logits_padded, atol=1e-5)


def test_load_pretrained_backbone_skips_mismatched_head(tmp_path):
    config = make_tiny_config()
    lm = GPT(config)
    ckpt_path = tmp_path / "pretrain.pt"
    torch.save({"model_state": lm.state_dict(), "config": config}, ckpt_path)

    clf = GPTClassifier(config, num_classes=5)
    result = clf.load_pretrained_backbone(ckpt_path)

    assert set(result["missing"]) == {"classifier.weight", "classifier.bias"}
    assert set(result["unexpected"]) == {"head.weight"}
    assert torch.equal(clf.tok_emb.weight, lm.tok_emb.weight)

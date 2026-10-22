import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import GPT, GPTConfig  # noqa: E402


def make_tiny_model() -> GPT:
    config = GPTConfig(vocab_size=50, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    return GPT(config)


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

"""1단계: 자모 단위 다음 토큰 예측으로 언어모델을 처음부터(랜덤 초기화) 학습한다.

data/raw/pretrain.jsonl의 모든 문서를 자모 토큰 스트림 하나로 이어붙이고,
block_size 길이로 잘라 "다음 토큰 맞추기" 과제를 반복 학습시킨다. 토크나이저
어휘도 이 코퍼스에서 직접 만든다(JamoTokenizer.build_vocab).

사용 (Kaggle GPU 권장):
    python scripts/train_pretrain.py \
        --data data/raw/pretrain.jsonl \
        --out checkpoints/pretrain.pt \
        --steps 3000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import GPT, GPTConfig  # noqa: E402
from tokenizer import JamoTokenizer  # noqa: E402


def encode_corpus(tokenizer: JamoTokenizer, data_path: Path) -> torch.Tensor:
    import json

    ids: list[int] = []
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line).get("text", "")
            if not text:
                continue
            ids.extend(tokenizer.encode(text, add_bos_eos=True))
    return torch.tensor(ids, dtype=torch.long)


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s : s + block_size] for s in starts])
    y = torch.stack([data[s + 1 : s + 1 + block_size] for s in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPT, data: torch.Tensor, block_size: int, batch_size: int, device: str, iters: int = 20
) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/raw/pretrain.jsonl"))
    parser.add_argument("--vocab-out", type=Path, default=Path("artifacts/vocab.json"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/pretrain.pt"))
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--min-freq", type=int, default=20, help="토크나이저 어휘에 넣을 기타 문자 최소 등장 횟수")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"토크나이저 어휘 구성 중... ({args.data})", file=sys.stderr)
    tokenizer = JamoTokenizer.build_vocab([args.data], min_freq=args.min_freq)
    args.vocab_out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.vocab_out)
    print(f"어휘 크기: {tokenizer.vocab_size} (저장: {args.vocab_out})", file=sys.stderr)

    print("코퍼스 토큰화 중...", file=sys.stderr)
    data = encode_corpus(tokenizer, args.data)
    n_val = max(1, int(len(data) * args.val_fraction))
    train_data, val_data = data[:-n_val], data[-n_val:]
    print(f"전체 토큰 수: {len(data):,} (train {len(train_data):,} / val {len(val_data):,})", file=sys.stderr)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(args.device)
    print(f"모델 파라미터 수: {model.num_params():,} (device={args.device})", file=sys.stderr)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.block_size, args.batch_size, args.device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.steps:
            val_loss = estimate_loss(model, val_data, args.block_size, args.batch_size, args.device)
            elapsed = time.time() - t0
            print(
                f"step {step}/{args.steps}  train_loss {loss.item():.4f}  "
                f"val_loss {val_loss:.4f}  ({elapsed:.0f}s)",
                file=sys.stderr,
            )
            # 세션이 중간에 끊겨도(Kaggle idle disconnect 등) 최신 진행 상황을 건질 수 있도록
            # 매 평가 시점마다 같은 경로에 체크포인트를 덮어쓴다.
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state": model.state_dict(), "config": config, "step": step},
                args.out,
            )

    print(f"체크포인트 저장: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

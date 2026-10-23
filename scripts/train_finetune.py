"""2단계: 산업분야 분류 미세조정.

--pretrained로 1단계 체크포인트를 주면 그 백본을 이어받아 분류 head만 학습하고,
안 주면 같은 구조를 랜덤 초기화부터 바로 분류만 학습한다(비교용 베이스라인).
두 경우를 같은 vocab·같은 구조·같은 데이터 split·같은 시드로 돌려서 val 정확도를
비교하면, 사전학습이 실제로 얼마나 도움이 됐는지 숫자로 나온다.

사용:
    # 사전학습 백본으로 미세조정
    python scripts/train_finetune.py --pretrained checkpoints/pretrain.pt \
        --vocab artifacts/vocab.json --out checkpoints/finetune_pretrained.pt \
        --metrics-out artifacts/metrics_pretrained.json

    # 비교군: 사전학습 없이 랜덤 초기화부터 바로 분류만 학습
    python scripts/train_finetune.py --vocab artifacts/vocab.json \
        --out checkpoints/finetune_scratch.pt \
        --metrics-out artifacts/metrics_scratch.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import GPTClassifier, GPTConfig  # noqa: E402
from tokenizer import JamoTokenizer  # noqa: E402


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def encode_records(
    records: list[dict], tokenizer: JamoTokenizer, label_map: dict[str, int], max_len: int
) -> list[tuple[list[int], int]]:
    data = []
    for r in records:
        ids = tokenizer.encode(r["text"], add_bos_eos=False)[:max_len]
        if not ids:
            continue
        data.append((ids, label_map[r["category"]]))
    return data


def split_train_val(
    data: list[tuple[list[int], int]], val_fraction: float, seed: int
) -> tuple[list, list]:
    idx = list(range(len(data)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(idx) * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return [data[i] for i in train_idx], [data[i] for i in val_idx]


def iter_batches(
    data: list[tuple[list[int], int]],
    batch_size: int,
    pad_id: int,
    device: str,
    shuffle: bool,
    rng: random.Random,
):
    order = list(range(len(data)))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        batch = [data[i] for i in order[start : start + batch_size]]
        max_len = max(len(ids) for ids, _ in batch)
        x = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        lengths = torch.zeros(len(batch), dtype=torch.long)
        y = torch.zeros(len(batch), dtype=torch.long)
        for i, (ids, label) in enumerate(batch):
            x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            lengths[i] = len(ids)
            y[i] = label
        yield x.to(device), lengths.to(device), y.to(device)


@torch.no_grad()
def evaluate(model: GPTClassifier, data: list, batch_size: int, pad_id: int, device: str) -> dict:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    rng = random.Random(0)  # 평가는 섞을 필요 없지만 iter_batches 시그니처 통일용
    for x, lengths, y in iter_batches(data, batch_size, pad_id, device, shuffle=False, rng=rng):
        logits, loss = model(x, lengths, y)
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += len(y)
    model.train()
    return {"loss": loss_sum / total, "accuracy": correct / total, "n": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/raw/sector.jsonl"))
    parser.add_argument("--vocab", type=Path, required=True, help="1단계에서 저장한 vocab.json")
    parser.add_argument("--pretrained", type=Path, default=None, help="1단계 체크포인트 (안 주면 랜덤 초기화부터 시작)")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/finetune.pt"))
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4, help="분류 head 학습률")
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help="백본(tok_emb/pos_emb/blocks/ln_f) 학습률. "
        "--pretrained를 줬으면 기본값 lr/10 (사전학습 지식이 금방 덮어써지는 걸 방지), "
        "--pretrained 없으면 기본값 lr(=분류 head와 동일, 보호할 사전지식이 없으므로)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = JamoTokenizer.load(args.vocab)
    records = load_records(args.data)
    label_map = {label: i for i, label in enumerate(sorted({r["category"] for r in records}))}
    print(f"업종(라벨) 수: {len(label_map)}", file=sys.stderr)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )

    if args.pretrained:
        ckpt_config = torch.load(args.pretrained, map_location="cpu", weights_only=False)["config"]
        if ckpt_config.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                f"체크포인트 vocab_size({ckpt_config.vocab_size})와 "
                f"--vocab의 vocab_size({tokenizer.vocab_size})가 다릅니다 — "
                "1단계와 같은 vocab.json을 써야 합니다."
            )
        # 가중치 모양을 결정하는 필드만 체크포인트 걸 따르고, dropout처럼 모양과
        # 무관한 하이퍼파라미터는 이번 실행의 --dropout 값을 그대로 쓴다
        # (체크포인트 config를 통째로 덮어쓰면 --dropout이 조용히 무시됐었음).
        config = GPTConfig(
            vocab_size=ckpt_config.vocab_size,
            block_size=ckpt_config.block_size,
            n_layer=ckpt_config.n_layer,
            n_head=ckpt_config.n_head,
            n_embd=ckpt_config.n_embd,
            dropout=args.dropout,
        )

    if args.max_len > config.block_size:
        print(
            f"--max-len({args.max_len})이 모델 block_size({config.block_size})보다 커서 "
            f"{config.block_size}로 줄입니다.",
            file=sys.stderr,
        )
        args.max_len = config.block_size

    data = encode_records(records, tokenizer, label_map, args.max_len)
    train_data, val_data = split_train_val(data, args.val_fraction, args.seed)
    print(f"train {len(train_data)} / val {len(val_data)}", file=sys.stderr)

    model = GPTClassifier(config, num_classes=len(label_map)).to(args.device)

    if args.pretrained:
        result = model.load_pretrained_backbone(args.pretrained, device=args.device)
        print(f"사전학습 백본 로드: missing={result['missing']} unexpected={result['unexpected']}", file=sys.stderr)
    else:
        print("사전학습 없이 랜덤 초기화부터 시작 (베이스라인)", file=sys.stderr)

    print(f"모델 파라미터 수: {model.num_params():,} (device={args.device})", file=sys.stderr)

    if args.backbone_lr is not None:
        backbone_lr = args.backbone_lr
    elif args.pretrained:
        backbone_lr = args.lr / 10  # 사전학습 지식이 몇 epoch 안에 덮어써지지 않도록 천천히
    else:
        backbone_lr = args.lr  # 보호할 사전지식이 없으니 head와 같은 속도로 학습

    head_params = list(model.classifier.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]
    print(f"학습률: head={args.lr} backbone={backbone_lr}", file=sys.stderr)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": args.lr},
        ]
    )
    rng = random.Random(args.seed)

    best_val_acc = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        for x, lengths, y in iter_batches(
            train_data, args.batch_size, tokenizer.pad_id, args.device, shuffle=True, rng=rng
        ):
            _, loss = model(x, lengths, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate(model, train_data, args.batch_size, tokenizer.pad_id, args.device)
        val_metrics = evaluate(model, val_data, args.batch_size, tokenizer.pad_id, args.device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train_loss {train_metrics['loss']:.4f} train_acc {train_metrics['accuracy']:.3f}  "
            f"val_loss {val_metrics['loss']:.4f} val_acc {val_metrics['accuracy']:.3f}",
            file=sys.stderr,
        )

        if val_metrics["accuracy"] >= best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "label_map": label_map,
                    "epoch": epoch,
                    "val_accuracy": val_metrics["accuracy"],
                },
                args.out,
            )

    print(f"최고 val 정확도: {best_val_acc:.3f} (체크포인트: {args.out})", file=sys.stderr)

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(
            json.dumps(
                {
                    "pretrained": str(args.pretrained) if args.pretrained else None,
                    "best_val_accuracy": best_val_acc,
                    "n_train": len(train_data),
                    "n_val": len(val_data),
                    "n_classes": len(label_map),
                    "history": history,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

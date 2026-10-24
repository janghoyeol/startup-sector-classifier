"""2단계 미세조정 체크포인트로 업종을 예측하는 추론 래퍼.

체크포인트(모델 가중치 + config + 업종 라벨 맵)와 vocab.json을 로컬에서
못 찾으면, 환경변수로 지정된 Hugging Face Hub 레포에서 내려받아 캐싱한다
(체크포인트를 git에 직접 올리지 않기 위함 — corp-agent가 DART 고유번호
전체자료를 컨테이너 시작 시 캐싱하던 것과 같은 패턴).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPTClassifier  # noqa: E402
from tokenizer import JamoTokenizer  # noqa: E402

HF_REPO_ENV = "SECTOR_MODEL_HF_REPO"  # 예: "janghoyeol/startup-sector-classifier"


def _ensure_local(path: Path, hf_filename: str) -> Path:
    if path.exists():
        return path
    repo_id = os.environ.get(HF_REPO_ENV)
    if not repo_id:
        raise FileNotFoundError(
            f"{path}가 없고 {HF_REPO_ENV} 환경변수도 안 설정되어 있어 "
            "Hugging Face Hub에서 내려받을 수도 없습니다."
        )
    from huggingface_hub import hf_hub_download

    print(f"{path.name}을(를) 로컬에서 못 찾아 {repo_id}에서 내려받습니다...", file=sys.stderr)
    downloaded = hf_hub_download(repo_id=repo_id, filename=hf_filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(Path(downloaded).read_bytes())
    return path


class SectorPredictor:
    def __init__(
        self,
        checkpoint_path: Path | str = Path("checkpoints/finetune_pretrained.pt"),
        vocab_path: Path | str = Path("artifacts/vocab.json"),
        device: str = "cpu",
    ):
        checkpoint_path = _ensure_local(Path(checkpoint_path), "finetune_pretrained.pt")
        vocab_path = _ensure_local(Path(vocab_path), "vocab.json")

        self.device = device
        self.tokenizer = JamoTokenizer.load(vocab_path)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.config = ckpt["config"]
        self.label_map: dict[str, int] = ckpt["label_map"]
        self.id_to_label = {i: label for label, i in self.label_map.items()}

        self.model = GPTClassifier(self.config, num_classes=len(self.label_map)).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str, top_k: int = 5) -> list[dict]:
        text = text.strip()
        if not text:
            return []
        ids = self.tokenizer.encode(text, add_bos_eos=False)[: self.config.block_size]
        if not ids:
            return []
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(ids)], dtype=torch.long, device=self.device)

        logits, _ = self.model(x, lengths)
        probs = torch.softmax(logits[0], dim=-1)
        top = torch.topk(probs, min(top_k, probs.numel()))

        return [
            {"label": self.id_to_label[idx.item()], "probability": round(p.item(), 4)}
            for p, idx in zip(top.values, top.indices)
        ]

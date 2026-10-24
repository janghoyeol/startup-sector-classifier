"""산업분야 분류 FastAPI 서버.

CLI(scripts/)와 완전히 같은 SectorPredictor를 재사용한다 — 두 인터페이스가
서로 다른 답을 내놓을 일이 없다 (stat-agent/corp-agent와 같은 설계 원칙).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from predictor import SectorPredictor  # noqa: E402

MAX_INPUT_CHARS = 2000

app = FastAPI(title="startup-sector-classifier")
predictor: SectorPredictor | None = None


@app.on_event("startup")
def load_model() -> None:
    global predictor
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "checkpoints/finetune_pretrained.pt")
    vocab_path = os.environ.get("VOCAB_PATH", "artifacts/vocab.json")
    predictor = SectorPredictor(checkpoint_path=checkpoint_path, vocab_path=vocab_path)


class PredictRequest(BaseModel):
    text: str


@app.post("/api/predict")
def predict(req: PredictRequest):
    text = req.text[:MAX_INPUT_CHARS]
    predictions = predictor.predict(text, top_k=5)
    return {
        "predictions": predictions,
        "num_classes": len(predictor.label_map),
        "truncated": len(req.text) > MAX_INPUT_CHARS,
    }


@app.get("/healthz")
def healthz():
    return {"ok": predictor is not None}


STATIC_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=STATIC_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")

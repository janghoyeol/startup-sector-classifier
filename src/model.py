"""처음부터 설계한 작은 GPT 스타일 디코더 전용 트랜스포머.

HuggingFace transformers 같은 라이브러리로 기존 모델을 불러오지 않는다 —
여기 있는 건 PyTorch의 기본 레이어(nn.Linear, nn.Embedding, nn.LayerNorm)만
조합해서 직접 쌓은 구조다: 토큰 임베딩 + 위치 임베딩 → (셀프 어텐션 + MLP)
블록을 n_layer번 반복 → 최종 선형 출력층.

구조 자체는 GPT-2와 같은 계열(learned positional embedding, LayerNorm,
GELU MLP)이다 — "완전히 새로운 아키텍처를 발명"하는 게 목표가 아니라,
가중치를 랜덤 초기화부터 직접 학습시키는 "from scratch" 원칙을 지키는
것이 핵심이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # 미래 토큰을 보지 못하게 막는 하삼각 마스크 (causal / autoregressive)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-norm + residual: 각 서브레이어를 거치기 전에 LayerNorm, 결과는 더해서 누적
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 입력 임베딩과 출력층 가중치 공유 (weight tying) — 파라미터 절약, 흔한 관행
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.config.block_size, "입력 길이가 block_size를 넘었습니다"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


class GPTClassifier(nn.Module):
    """1단계에서 사전학습한 GPT 백본 위에 분류 head를 얹은 모델 (2단계 미세조정용).

    필드 이름(tok_emb/pos_emb/drop/blocks/ln_f)을 `GPT`와 똑같이 맞춰뒀다 — 그래야
    1단계가 저장한 체크포인트의 state_dict를 `strict=False`로 그대로 얹을 수 있다.
    출력층만 모양이 다르므로(vocab_size 대신 num_classes) 자동으로 빠지고 새로
    랜덤 초기화된 채 남는다. 사전학습을 안 쓰고 싶으면 그냥 `load_pretrained_backbone`을
    안 부르면 되고, 그러면 이 클래스 전체가 랜덤 초기화부터 시작한다 — 두 경우를
    같은 코드·같은 구조로 비교할 수 있게 하려는 설계다.
    """

    def __init__(self, config: GPTConfig, num_classes: int):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.classifier = nn.Linear(config.n_embd, num_classes)
        self.apply(GPT._init_weights)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def load_pretrained_backbone(self, checkpoint_path, device: str = "cpu") -> dict:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        missing, unexpected = self.load_state_dict(ckpt["model_state"], strict=False)
        return {"missing": missing, "unexpected": unexpected}

    def forward(
        self,
        idx: torch.Tensor,
        lengths: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.config.block_size, "입력 길이가 block_size를 넘었습니다"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # 각 예제에서 "마지막 실제 토큰" 위치의 hidden state를 뽑아 분류에 쓴다.
        # causal(단방향) attention이라, 그 뒤에 붙는 패딩 토큰이 이 표현에 영향을
        # 주지 않는다 — 별도의 padding mask 없이도 안전하다.
        last_idx = (lengths - 1).clamp(min=0)
        pooled = x[torch.arange(B, device=idx.device), last_idx]
        logits = self.classifier(pooled)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

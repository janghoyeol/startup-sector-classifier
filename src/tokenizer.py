"""자모 단위 토크나이저.

한글 음절(가, 힣 등)은 초성·중성·종성의 조합이다. 유니코드가 이미 이 분해를
"정준 분해(NFD)"로 표준화해뒀기 때문에, 직접 산술 공식을 짤 필요 없이
`unicodedata.normalize("NFD", text)`만 거치면 음절이 자동으로 자모 시퀀스로
풀린다 (복원은 반대로 NFC). 그 결과를 한 글자씩 토큰화한다.

이렇게 하면 어휘 수가 (초성 19 + 중성 21 + 종성 27 + 특수토큰 4 + 코퍼스에서
관찰된 기타 문자) 수준으로, 수만 개짜리 BPE 어휘보다 훨씬 작다. 코퍼스에
없던 조합(신조어·오타)도 자모 단위로는 항상 표현 가능하다는 게 핵심 설계
근거다 — docs/decisions/0002-jamo-tokenizer.md 참고.

기타 문자(영문/숫자/문장부호/한자 등)는 코퍼스에서 일정 빈도 이상 등장한
것만 어휘에 넣고, 나머지는 [UNK]로 처리한다 — 무작정 다 담으면 자모 단위로
줄인 의미가 없어진다.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path

PAD, UNK, BOS, EOS = "[PAD]", "[UNK]", "[BOS]", "[EOS]"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]

# 유니코드 Hangul Jamo 블록의 결합형(conjoining) 자모 — NFD가 음절을 분해하면
# 항상 이 범위 안의 문자로 떨어진다.
CHOSEONG = [chr(c) for c in range(0x1100, 0x1113)]  # 초성 19개
JUNGSEONG = [chr(c) for c in range(0x1161, 0x1176)]  # 중성 21개
JONGSEONG = [chr(c) for c in range(0x11A8, 0x11C3)]  # 종성 27개 (받침 없으면 그냥 안 나옴)
HANGUL_JAMO = CHOSEONG + JUNGSEONG + JONGSEONG  # 항상 어휘에 포함(67개)


class JamoTokenizer:
    def __init__(self, vocab: list[str]):
        self.id_to_token = list(vocab)
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        self.unk_id = self.token_to_id[UNK]
        self.pad_id = self.token_to_id[PAD]
        self.bos_id = self.token_to_id[BOS]
        self.eos_id = self.token_to_id[EOS]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @staticmethod
    def _decompose(text: str) -> str:
        return unicodedata.normalize("NFD", text)

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        chars = self._decompose(text)
        ids = [self.token_to_id.get(ch, self.unk_id) for ch in chars]
        if add_bos_eos:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        chars = [
            self.id_to_token[i]
            for i in ids
            if i not in (self.pad_id, self.bos_id, self.eos_id)
        ]
        return unicodedata.normalize("NFC", "".join(chars))

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(self.id_to_token, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "JamoTokenizer":
        vocab = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(vocab)

    @classmethod
    def build_vocab(
        cls,
        jsonl_paths: list[Path],
        text_field: str = "text",
        min_freq: int = 20,
    ) -> "JamoTokenizer":
        counts: Counter[str] = Counter()
        for path in jsonl_paths:
            with Path(path).open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = obj.get(text_field, "")
                    counts.update(cls._decompose(text))

        other = sorted(ch for ch, n in counts.items() if n >= min_freq and ch not in HANGUL_JAMO)
        vocab = SPECIAL_TOKENS + HANGUL_JAMO + other
        return cls(vocab)

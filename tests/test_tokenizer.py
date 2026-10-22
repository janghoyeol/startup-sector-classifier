import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenizer import JamoTokenizer, SPECIAL_TOKENS, HANGUL_JAMO  # noqa: E402


def make_tokenizer(extra_chars: str = "") -> JamoTokenizer:
    vocab = SPECIAL_TOKENS + HANGUL_JAMO + sorted(set(extra_chars))
    return JamoTokenizer(vocab)


def test_roundtrip_basic_korean():
    tok = make_tokenizer(" .,()~0123456789가-힣abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    text = "안녕하세요. 반갑습니다 (hello) 123"
    assert tok.decode(tok.encode(text)) == text


def test_vocab_always_has_all_jamo():
    tok = make_tokenizer()
    assert tok.vocab_size == len(SPECIAL_TOKENS) + len(HANGUL_JAMO)


def test_unknown_char_maps_to_unk_and_is_lossy():
    tok = make_tokenizer()  # "!" 같은 기호는 어휘에 없음
    ids = tok.encode("가나다!")
    assert tok.unk_id in ids
    # UNK는 원래 글자를 복원 못 함 (정보 손실이 자연스러운 동작)
    assert tok.decode(ids) != "가나다!"


def test_bos_eos_added_and_stripped_on_decode():
    tok = make_tokenizer()
    ids = tok.encode("가나", add_bos_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert tok.decode(ids) == "가나"


def test_build_vocab_from_corpus(tmp_path):
    jsonl = tmp_path / "corpus.jsonl"
    jsonl.write_text(
        '{"text": "가나다 가나다 가나다"}\n{"text": "희귀문자@"}\n',
        encoding="utf-8",
    )
    tok = JamoTokenizer.build_vocab([jsonl], min_freq=2)
    # "가나다"가 3번 반복돼 등장 빈도 min_freq(2) 이상인 자모만 어휘에 추가되고,
    # 단 한 번 나온 "@" 같은 희귀 문자는 제외된다.
    assert "@" not in tok.id_to_token
    assert tok.decode(tok.encode("가나다 가나다 가나다")) == "가나다 가나다 가나다"

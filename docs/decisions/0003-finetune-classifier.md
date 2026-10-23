# 0003. 2단계 미세조정 — 분류 head + 사전학습 유무 비교

## 상태
승인됨 (2026-10-23)

## 결정
`GPTClassifier`를 1단계 `GPT`와 필드 이름(tok_emb/pos_emb/blocks/ln_f)이 같게
설계해서, 1단계 체크포인트를 `load_state_dict(strict=False)`로 그대로 얹을 수
있게 했다. 출력층만 모양이 다르므로(vocab_size → num_classes) 자동으로 빠지고
새로 초기화된다.

`scripts/train_finetune.py`는 `--pretrained` 인자로 두 가지 실행을 모두
지원한다:
- 체크포인트를 주면 사전학습 백본으로 시작
- 안 주면 같은 구조·같은 vocab·같은 데이터 split·같은 시드로 **랜덤
  초기화부터** 분류만 학습 (베이스라인)

이렇게 하면 "사전학습이 실제로 도움이 됐는가"를 같은 조건에서 val 정확도로
직접 비교할 수 있다 — 이게 이번 프로젝트의 핵심 평가 포인트다.

## 풀링 방식
각 예제에서 "마지막 실제 토큰" 위치의 hidden state를 뽑아 분류에 쓴다
(BERT의 [CLS] 토큰 같은 별도 토큰을 추가하는 대신). causal(단방향)
self-attention이라 그 뒤에 오는 패딩 토큰이 앞쪽 표현에 전혀 영향을 주지
않으므로, 별도의 attention mask 구현 없이도 안전하다 —
`tests/test_model.py::test_classifier_padding_after_last_real_token_does_not_change_pooled_output`
로 검증.

## 알려진 제약
- 라벨 데이터가 500개뿐이라 업종(라벨) 수(21개) 대비 클래스당 평균 24개
  수준 — 클래스 불균형을 별도로 보정하지 않음. 결과 해석 시 감안해야 함.
- 학습률을 백본/분류head 구분 없이 동일하게 적용 — 보통은 사전학습된
  백본에 더 작은 학습률을 쓰는 게 정석이지만, 1차 버전에서는 단순하게
  시작하고 결과 보고 필요하면 개선.

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
- 라벨 데이터가 500~2000개 수준이라 업종(라벨) 수(21개) 대비 클래스당
  평균 개수가 적음 — 클래스 불균형을 별도로 보정하지 않음. 결과 해석 시
  감안해야 함.

## 추가 결정 (2026-10-23): 백본/head 차등 학습률
첫 비교 실험(500개, 라벨 4개 — 버그로 인한 것) 결과 사전학습 80.0% vs
스크래치 81.3%, 데이터를 2000개/21개 라벨로 다시 모은 두 번째 실험에서도
**둘 다 정확히 23.4%로 동률**이었다. 두 실험 다 train_acc가 97%+까지
암기하는 동안 val_loss는 계속 치솟는 심한 과적합 패턴을 보였다.

원인으로 지목한 것: 백본과 분류 head를 구분 없이 같은 학습률(3e-4)로
학습시키고 있었다 — 이러면 사전학습으로 얻은 표현이 몇 epoch 안에
덮어써져서, 사전학습을 했든 안 했든 결과가 수렴한다. `--backbone-lr`을
추가해 기본값을 다음과 같이 분리했다:
- `--pretrained` 있음 → 백본 lr = head lr의 1/10 (사전지식 보존)
- `--pretrained` 없음 → 백본 lr = head lr과 동일 (보호할 사전지식이 없음)

동시에 `--pretrained`를 줬을 때 체크포인트의 `config`를 통째로 덮어써서
`--dropout` CLI 인자가 조용히 무시되던 버그도 같이 고쳤다 — vocab_size/
block_size/n_layer/n_head/n_embd처럼 가중치 모양을 결정하는 필드만
체크포인트에서 가져오고, dropout은 이번 실행 인자를 그대로 쓴다.

재실험 결과는 다음 턴에 기록 예정.

# 0005. 서빙 구조와 배포

## 상태
승인됨 (2026-10-24)

## 결정

### CLI와 웹이 같은 추론 코드를 쓴다
`src/predictor.py`의 `SectorPredictor`를 FastAPI(`web/api.py`)가 그대로
가져다 쓴다. stat-agent/corp-agent 때와 같은 원칙 — CLI와 웹 인터페이스가
서로 다른 추론 경로를 타면 두 군데가 다른 답을 낼 수 있어서, 아예 하나만
존재하게 한다.

### 체크포인트는 git이 아니라 Hugging Face Hub에서
미세조정 체크포인트(`finetune_pretrained.pt`, 약 48MB)와 `vocab.json`은
레포에 커밋하지 않는다(`.gitignore`로 계속 제외). 대신 `SectorPredictor`가
로컬에 없으면 `SECTOR_MODEL_HF_REPO` 환경변수가 가리키는 Hugging Face
Hub 레포에서 내려받아 캐싱한다 — corp-agent가 컨테이너 시작 시 DART
고유번호 전체자료를 캐싱하던 것과 같은 패턴.

GitHub에 커밋하기엔(100MB 제한에 걸리진 않지만) 바이너리를 git 히스토리에
쌓는 게 좋지 않고, Hugging Face Hub는 모델 아티팩트를 위해 만들어진
곳이라 더 적합하다고 판단.

### Docker 이미지에서 torch는 CPU 전용 wheel로 설치
기본 `pip install torch`는 CUDA 포함 wheel(수 GB)을 받는다. Render
서버는 GPU가 없으니 `--index-url https://download.pytorch.org/whl/cpu`로
CPU 전용 wheel을 따로 설치해 이미지 크기와 빌드 시간을 줄였다.

### 모델 규모와 무료 플랜 호환성
체크포인트가 약 48MB(파라미터 1,186만 개, fp32)로, 애초에 걱정했던
"200MB짜리를 무료로 어떻게 서빙하냐" 수준에는 전혀 못 미친다. Render
무료 플랜 RAM(보통 512MB)에 여유 있게 들어간다 — 추가 양자화 등은
불필요.

## 알려진 제약
- 체크포인트 다운로드는 컨테이너가 깨어날 때마다(무료 플랜은 15분 비활성 시
  슬립) 매번 일어나는 게 아니라, 컨테이너 디스크에 캐싱되면 재다운로드하지
  않는다 — 단, Render 무료 플랜은 재배포/재시작 시 디스크가 초기화될 수
  있어 그 경우 다시 받는다. 정상 동작이지만 첫 요청이 느릴 수 있다는 점은
  showcase 쪽 "깨우기" UI로 이미 사용자에게 안내 중인 패턴과 같다.

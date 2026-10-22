"""한국어 위키백과 전체 덤프를 내려받아 사전학습용 pretrain.jsonl을 만든다.

API로 문서를 한 건씩 모으는 방식은 429 속도 제한 때문에 현실적으로 수천 건 이상을
모으기 어렵고, 애초에 언어모델 사전학습에는 그 정도 규모로는 턱없이 부족하다(최소
수백만~수억 토큰 단위 텍스트가 필요). 그래서 위키미디어가 공식 배포하는 전체 덤프
파일을 통째로 받아 쓴다 — 요청 횟수 제한이 없고, 한국어 위키백과 전체가 압축
상태로 약 1.3GB라 다루기 쉬운 크기다.

단계:
    1. kowiki-latest-pages-articles.xml.bz2 다운로드 (아직 없으면)
    2. wikiextractor로 위키 마크업(템플릿/표/인포박스)을 걷어내고 순수 텍스트만 추출
    3. 추출된 문서 중 --max-articles 개(랜덤 셔플 후)만 뽑아 pretrain.jsonl로 저장
    4. 중간 산출물(덤프 원본, 추출 폴더)은 --keep-raw 안 주면 정리해서 디스크 절약

사용 (Kaggle/로컬, 몇 GB 디스크와 몇 분~수십 분 소요):
    pip install wikiextractor
    python scripts/build_pretrain_corpus.py --max-articles 50000 --out data/raw/pretrain.jsonl
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import requests

DUMP_URL = "https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2"


def download_dump(dest: Path) -> None:
    if dest.exists():
        print(f"덤프 이미 있음: {dest} ({dest.stat().st_size / 1e6:.0f} MB), 다운로드 건너뜀", file=sys.stderr)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"덤프 다운로드 중... ({DUMP_URL})", file=sys.stderr)
    with requests.get(DUMP_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r{done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", file=sys.stderr)
        print(file=sys.stderr)
        tmp.rename(dest)


def run_wikiextractor(dump_path: Path, out_dir: Path, processes: int) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"추출 결과 이미 있음: {out_dir}, 추출 건너뜀", file=sys.stderr)
        return
    print("wikiextractor로 텍스트 추출 중 (시간이 꽤 걸립니다)...", file=sys.stderr)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "wikiextractor.WikiExtractor",
            str(dump_path),
            "-o",
            str(out_dir),
            "--json",
            "--processes",
            str(processes),
        ],
        check=True,
    )


def build_pretrain_jsonl(extracted_dir: Path, out_path: Path, max_articles: int, min_chars: int, seed: int) -> int:
    records = []
    for path in extracted_dir.rglob("wiki_*"):
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "").strip()
                if len(text) < min_chars:
                    continue
                records.append({"title": obj.get("title"), "text": text})

    print(f"조건을 만족하는 문서 수: {len(records)}", file=sys.stderr)
    rng = random.Random(seed)
    rng.shuffle(records)
    records = records[:max_articles]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-articles", type=int, default=50000, help="최종 pretrain.jsonl에 담을 문서 수")
    parser.add_argument("--min-chars", type=int, default=200, help="너무 짧은 문서(스텁) 제외 기준")
    parser.add_argument("--out", type=Path, default=Path("data/raw/pretrain.jsonl"))
    parser.add_argument("--workdir", type=Path, default=Path("data/_wiki_dump_work"))
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-raw", action="store_true", help="덤프 원본/추출 폴더를 정리하지 않고 남겨둠")
    args = parser.parse_args()

    dump_path = args.workdir / "kowiki-latest-pages-articles.xml.bz2"
    extracted_dir = args.workdir / "extracted"

    download_dump(dump_path)
    run_wikiextractor(dump_path, extracted_dir, args.processes)
    n = build_pretrain_jsonl(extracted_dir, args.out, args.max_articles, args.min_chars, args.seed)
    print(f"완료: {args.out} 에 문서 {n}개 저장", file=sys.stderr)

    if not args.keep_raw:
        print("중간 산출물 정리 중...", file=sys.stderr)
        shutil.rmtree(args.workdir, ignore_errors=True)


if __name__ == "__main__":
    main()

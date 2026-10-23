"""한국어 위키백과 공식 API로 2단계 미세조정용 라벨 데이터를 모은다.

"분류:대한민국의 업종별 기업" 하위 카테고리(업종별)에 속한 회사 문서의 도입부 텍스트 +
카테고리명을 라벨로 수집한다.

(1단계 언어모델 사전학습용 대용량 텍스트는 API 건별 수집으로는 규모가 턱없이
부족해서 build_pretrain_corpus.py로 분리했다 — 위키백과 전체 덤프를 받아 쓴다.)

위키백과 API는 봇 차단이 없고 공식적으로 적절한 속도의 익명 사용을 허용한다
(https://www.mediawiki.org/wiki/API:Etiquette). 그래도 매 요청 사이 delay를 두고
429를 받으면 Retry-After를 존중해 재시도한다.

사용:
    python scripts/wiki_scrape.py sector --limit 500 --out data/raw/sector.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

API = "https://ko.wikipedia.org/w/api.php"
USER_AGENT = (
    "startup-sector-classifier/0.1 "
    "(portfolio research project; contact: janghoyeol1@gmail.com)"
)
ROOT_CATEGORY = "분류:대한민국의 업종별 기업"


def call(session: requests.Session, params: dict, delay: float) -> dict:
    params = {**params, "format": "json"}
    for attempt in range(5):
        resp = session.get(API, params=params, timeout=20)
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", 10 * (attempt + 1)))
            print(f"429, {wait}s 대기...", file=sys.stderr)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        time.sleep(delay)
        return resp.json()
    raise RuntimeError("반복적으로 429를 받아 중단합니다")


def list_category_members(
    session: requests.Session, category: str, cmtype: str, delay: float
) -> list[str]:
    titles: list[str] = []
    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": cmtype,
            "cmlimit": "500",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = call(session, params, delay)
        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members)
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
    return titles


def fetch_intro(session: requests.Session, title: str, delay: float) -> str | None:
    data = call(
        session,
        {
            "action": "query",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "titles": title,
        },
        delay,
    )
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        text = page.get("extract", "").strip()
        if text:
            return text
    return None


def already_done(out_path: Path, key: str = "title") -> set[str]:
    if not out_path.exists():
        return set()
    seen = set()
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)[key])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def cmd_sector(args: argparse.Namespace) -> None:
    import random

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    print("하위 카테고리(업종) 목록 수집 중...", file=sys.stderr)
    subcats = list_category_members(session, ROOT_CATEGORY, "subcat", args.delay)
    print(f"업종 카테고리 수: {len(subcats)}", file=sys.stderr)

    seen = already_done(args.out)
    print(f"이미 수집됨: {len(seen)}개, 건너뜀", file=sys.stderr)

    # 카테고리별 소속 문서 목록을 먼저 다 모아둔다 (아직 본문은 안 가져옴 — 가벼움).
    # 한 카테고리를 다 소진한 뒤에야 다음 카테고리로 넘어가면, 앞쪽 카테고리 하나가
    # 문서가 많을 경우 --limit을 혼자 다 채워버려서 다른 업종이 하나도 안 들어갈 수
    # 있다(실제로 그랬다 — 21개 카테고리인데 4개만 나왔음). 그래서 카테고리별로
    # 큐를 만들어두고 한 바퀴씩 돌아가며(round-robin) 하나씩 뽑는다.
    print("카테고리별 문서 목록 수집 중...", file=sys.stderr)
    rng = random.Random(args.seed)
    queues: dict[str, list[str]] = {}
    for category in subcats:
        pages = list_category_members(session, category, "page", args.delay)
        rng.shuffle(pages)
        label = category.removeprefix("분류:")
        queues[label] = pages
        print(f"  {label}: {len(pages)}개", file=sys.stderr)

    per_category_cap = args.per_category_cap or max(1, -(-args.limit // max(1, len(queues))))
    print(f"카테고리당 최대 {per_category_cap}개까지 균형 있게 수집", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    collected = len(seen)
    per_category_count: dict[str, int] = {label: 0 for label in queues}
    labels_cycle = [label for label in queues if queues[label]]

    with args.out.open("a", encoding="utf-8") as f, tqdm(
        total=args.limit, initial=collected, desc="회사 문서 수집"
    ) as pbar:
        while collected < args.limit and labels_cycle:
            next_cycle = []
            for label in labels_cycle:
                if collected >= args.limit:
                    break
                queue = queues[label]
                if per_category_count[label] >= per_category_cap or not queue:
                    continue
                pbar.set_postfix_str(label[:20])
                title = queue.pop()
                if title in seen:
                    next_cycle.append(label)
                    continue
                text = fetch_intro(session, title, args.delay)
                if text:
                    f.write(
                        json.dumps({"title": title, "category": label, "text": text}, ensure_ascii=False)
                        + "\n"
                    )
                    f.flush()
                    seen.add(title)
                    collected += 1
                    per_category_count[label] += 1
                    pbar.update(1)
                if queue and per_category_count[label] < per_category_cap:
                    next_cycle.append(label)
            labels_cycle = next_cycle

    print("업종별 최종 수집 개수:", file=sys.stderr)
    for label, n in sorted(per_category_count.items(), key=lambda kv: -kv[1]):
        if n:
            print(f"  {label}: {n}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_sector = sub.add_parser("sector", help="업종 라벨 포함 회사 문서 수집")
    p_sector.add_argument("--limit", type=int, default=500)
    p_sector.add_argument("--out", type=Path, default=Path("data/raw/sector.jsonl"))
    p_sector.add_argument("--delay", type=float, default=1.0)
    p_sector.add_argument("--seed", type=int, default=42)
    p_sector.add_argument(
        "--per-category-cap",
        type=int,
        default=None,
        help="카테고리당 최대 수집 개수 (기본값: limit을 카테고리 수로 나눠서 자동 계산)",
    )
    p_sector.set_defaults(func=cmd_sector)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

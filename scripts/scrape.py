"""thevc.kr 공개 프로필 페이지에서 회사 소개글 + 업종 태그를 수집한다.

robots.txt(https://thevc.kr/robots.txt)는 `/api/`만 금지하고 일반 UA는 허용하지만,
실제로는 AWS WAF가 "나는 봇입니다"식 User-Agent는 403으로 막고 일반 브라우저처럼
보이는 요청만 통과시킨다(robots.txt 정책과 WAF 동작이 서로 모순됨 — 확인 후 사용자
승인받음, docs/decisions/0001-data-source-and-scraping-policy.md 참고). 그래서
이 스크립트는 평범한 Chrome User-Agent + 표준 Accept 헤더를 쓴다.

각 페이지의 JSON-LD(<script type="application/ld+json">, @graph 안의 Organization
노드)에 name/description/knowsAbout/keywords가 이미 구조화되어 있어, 별도의 HTML
파싱(칩 긁기) 없이 그대로 쓴다.

사용:
    python scripts/scrape.py --limit 200 --out data/raw/companies.jsonl
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from tqdm import tqdm

BASE = "https://thevc.kr"
SITEMAP_INDEX = f"{BASE}/sitemap_index.xml"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}
LD_JSON_RE = re.compile(r'<script type="application/ld\+json"[^>]*>(.*?)</script>', re.S)
LOC_RE = re.compile(r"<loc>([^<]+)</loc>")


def get(session: requests.Session, url: str, timeout: float = 20) -> str | None:
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            return None
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return None


def list_profile_urls(session: requests.Session) -> list[str]:
    index_xml = get(session, SITEMAP_INDEX)
    if not index_xml:
        raise RuntimeError("사이트맵 인덱스를 가져오지 못했습니다")
    shard_urls = [u for u in LOC_RE.findall(index_xml) if "profile" in u]

    all_urls: list[str] = []
    for shard_url in shard_urls:
        xml = get(session, shard_url)
        if not xml:
            continue
        all_urls.extend(LOC_RE.findall(xml))
        time.sleep(0.5)
    return all_urls


def extract_organization(html: str) -> dict | None:
    for blob in LD_JSON_RE.findall(html):
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        nodes = data.get("@graph", [data]) if isinstance(data, dict) else []
        for node in nodes:
            if isinstance(node, dict) and node.get("@type") == "Organization":
                if "description" in node or "keywords" in node:
                    return node
    return None


def already_scraped(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    seen = set()
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["url"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200, help="새로 수집할 회사 수")
    parser.add_argument("--out", type=Path, default=Path("data/raw/companies.jsonl"))
    parser.add_argument("--delay", type=float, default=1.2, help="요청 간 최소 대기(초)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)

    print("사이트맵에서 프로필 URL 목록 수집 중...", file=sys.stderr)
    all_urls = list_profile_urls(session)
    print(f"전체 프로필 수: {len(all_urls)}", file=sys.stderr)

    rng = random.Random(args.seed)
    rng.shuffle(all_urls)

    seen = already_scraped(args.out)
    print(f"이미 수집됨: {len(seen)}개, 건너뜀", file=sys.stderr)

    todo = [u for u in all_urls if u not in seen][: args.limit]

    with args.out.open("a", encoding="utf-8") as f:
        for url in tqdm(todo, desc="scraping"):
            html = get(session, url)
            time.sleep(args.delay + rng.uniform(0, 0.4))
            if not html:
                continue
            org = extract_organization(html)
            if not org or not org.get("description"):
                continue
            record = {
                "url": url,
                "name": org.get("name"),
                "description": org.get("description"),
                "knows_about": org.get("knowsAbout", []),
                "keywords": org.get("keywords", []),
                "founding_date": org.get("foundingDate"),
                "region": (org.get("address") or {}).get("addressRegion"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

from .config import MANIFEST, SNAPSHOTS, read_study, second_tuesdays, write_json

USER_AGENT = "TSFM-Prospective-CPU-Research/0.1 (reproducible academic evaluation)"
MODELS = {
    "chronos_bolt_tiny": "amazon/chronos-bolt-tiny",
    "chronos_t5_tiny": "amazon/chronos-t5-tiny",
    "chronos2_small": "autogluon/chronos-2-small",
    "ttm_512_week_context": "ibm-granite/granite-timeseries-ttm-r2",
}


class _GitHubSearchParser(HTMLParser):
    """Read repository names/star counts from GitHub's star-sorted public search page."""
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_list = False
        self.current: dict[str, Any] | None = None
        self.rows: list[dict[str, Any]] = []

    @staticmethod
    def _star_count(label: str) -> int | None:
        match = re.fullmatch(r"([\d,.]+)([kKmM]?) stars", label.strip())
        if not match:
            return None
        value = float(match.group(1).replace(",", ""))
        suffix = match.group(2).lower()
        return int(value * (1_000 if suffix == "k" else 1_000_000 if suffix == "m" else 1))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        classes = attr.get("class") or ""
        if attr.get("data-testid") == "results-list":
            self.in_list = True
        if not self.in_list:
            return
        if self.current is None and "Result-module__Result__" in classes:
            self.current = {"depth": 1, "full_name": None, "stars": None}
            return
        if self.current is None:
            return
        if tag == "div":
            self.current["depth"] += 1
        href = attr.get("href") or ""
        if tag == "a" and self.current["full_name"] is None and re.fullmatch(r"/[^/]+/[^/]+", href):
            self.current["full_name"] = href.strip("/")
        count = self._star_count(attr.get("aria-label") or "")
        if count is not None:
            self.current["stars"] = count
        language = re.fullmatch(r"(.+?) language", attr.get("aria-label") or "")
        if language:
            self.current["language"] = language.group(1)

    def handle_endtag(self, tag: str) -> None:
        if self.current is None or tag != "div":
            return
        self.current["depth"] -= 1
        if self.current["depth"] == 0:
            self.rows.append(self.current)
            self.current = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def save_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": str(path), "bytes": len(payload), "sha256": sha256_bytes(payload)}


def request_with_retry(url: str, *, headers: dict[str, str] | None = None,
                       attempts: int = 6, timeout: int = 60,
                       stats_202_is_retryable: bool = False) -> requests.Response:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    if headers:
        session.headers.update(headers)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(url, timeout=timeout)
            retryable = response.status_code in (408, 429, 500, 502, 503, 504)
            retryable |= response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0"
            retryable |= stats_202_is_retryable and response.status_code == 202
            if not retryable:
                response.raise_for_status()
                return response
            last_error = requests.HTTPError(f"HTTP {response.status_code} from {url}", response=response)
            retry_after = response.headers.get("Retry-After")
        except requests.RequestException as exc:
            last_error = exc
            retry_after = None
        if attempt + 1 < attempts:
            if retry_after and retry_after.isdigit():
                wait = min(120, int(retry_after))
            elif last_error and getattr(last_error, "response", None) is not None and last_error.response.headers.get("X-RateLimit-Reset"):
                reset = int(last_error.response.headers["X-RateLimit-Reset"])
                wait = min(120, max(1, reset - int(time.time()) + 1))
            else:
                wait = min(30, 2 ** attempt)
            time.sleep(wait)
    assert last_error is not None
    raise last_error


def discover_repositories(destination: Path) -> dict[str, list[dict[str, Any]]]:
    study = read_study()
    per_language = int(study["domains"]["github"]["repositories_per_language"])
    cohort: dict[str, list[dict[str, Any]]] = {}
    for language in study["domains"]["github"]["languages"]:
        query = f"language:{language} stars:>10000"
        url = "https://github.com/search"
        # The REST search resource is frequently exhausted on unauthenticated shared IPs. The
        # public GitHub search page supports the same language filters and explicit star order.
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/136 Safari/537.36"})
        chosen: list[dict[str, Any]] = []
        response = None
        response_body = b""
        raw_rows = 0
        for page in range(1, 4):
            for attempt in range(4):
                try:
                    response = session.get(url, params={"q": query, "type": "repositories", "s": "stars",
                                                        "o": "desc", "p": page}, timeout=45)
                    response.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == 3:
                        raise
                    time.sleep(min(30, 2 ** attempt))
            assert response is not None
            response_body += response.content
            save_bytes(destination / "github_search" / f"{language.lower()}_page{page}.html", response.content)
            parser = _GitHubSearchParser()
            parser.feed(response.text)
            raw_rows += len(parser.rows)
            for result_rank, item in enumerate(parser.rows, start=1):
                full_name = item.get("full_name")
                if not full_name or item.get("stars") is None or item.get("language") != language:
                    continue
                owner, repo_name = full_name.split("/", 1)
                try:
                    metadata = request_with_retry(
                        f"https://api.github.com/repos/{owner}/{repo_name}", attempts=3, timeout=20,
                        headers={"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
                    info = metadata.json()
                    save_bytes(destination / "github_repos" / f"{full_name.replace('/', '__')}.json", metadata.content)
                except requests.RequestException:
                    continue
                if info.get("archived") or info.get("fork") or info.get("language") != language:
                    continue
                chosen.append({"full_name": full_name, "language": language,
                               "stars_at_freeze": info["stargazers_count"],
                               "rank_at_freeze": len(chosen) + 1,
                               "search_result_rank": (page - 1) * 10 + result_rank,
                               "html_url": "https://github.com/" + full_name, "repo_id": info["id"]})
                if len(chosen) == per_language:
                    break
            if len(chosen) == per_language:
                break
        if len(chosen) != per_language:
            raise RuntimeError(f"GitHub sorted search returned only {len(chosen)} eligible {language} repositories")
        cohort[language] = chosen
        cohort[language + "__query"] = [{"query": query, "sort": "stars", "order": "desc",
                                          "selection_source": "github.com/search HTML",
                                          "status_code": response.status_code,
                                          "fetched_at_utc": utc_now(),
                                          "response_sha256": sha256_bytes(response_body),
                                          "result_rows": raw_rows }]
    return cohort


def resolve_model_revisions() -> dict[str, dict[str, str]]:
    out = {}
    for name, repo_id in MODELS.items():
        if name == "ttm_512_week_context":
            response = request_with_retry(f"https://huggingface.co/api/models/{repo_id}/refs", attempts=4)
            refs = response.json().get("branches", [])
            weekly_ref = next((ref for ref in refs if ref.get("name") == "512-48-ft-r2.1"), None)
            if weekly_ref is None:
                raise RuntimeError("Pinned TTM weekly branch 512-48-ft-r2.1 is unavailable")
            revision = weekly_ref["targetCommit"]
            branch_name = weekly_ref["name"]
        else:
            response = request_with_retry(f"https://huggingface.co/api/models/{repo_id}", attempts=4)
            revision = response.json()["sha"]
            branch_name = "main"
        out[name] = {"repo_id": repo_id, "revision": revision, "branch_name": branch_name,
                     "resolved_at_utc": utc_now(), "api_response_sha256": sha256_bytes(response.content)}
    return out


def download_ieso_snapshot(destination: Path, through_year: int | None = None) -> dict[str, Any]:
    study = read_study()
    base = study["domains"]["ieso"]["source"].rsplit("/", 1)[0]
    through_year = through_year or datetime.now().year
    records: list[dict[str, Any]] = []
    for year in range(2002, through_year + 1):
        url = f"{base}/PUB_Demand_{year}.csv"
        response = request_with_retry(url, attempts=5)
        if "csv" not in response.headers.get("Content-Type", "").lower():
            raise ValueError(f"IESO returned non-CSV content for {year}: {response.headers.get('Content-Type')}")
        record = save_bytes(destination / "ieso" / f"PUB_Demand_{year}.csv", response.content)
        record.update({"year": year, "url": url, "fetched_at_utc": utc_now(),
                       "last_modified": response.headers.get("Last-Modified")})
        records.append(record)
    return {"source": "IESO PUB_Demand annual CSV", "files": records}


def download_github_activity(destination: Path, repositories: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for repo in repositories:
        owner, name = repo["full_name"].split("/", 1)
        url = f"https://api.github.com/repos/{owner}/{name}/stats/commit_activity"
        response = None
        try:
            response = request_with_retry(
                url,
                headers={"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"},
                attempts=4,
                timeout=25,
                stats_202_is_retryable=True,
            )
            raw = response.content
            status = response.status_code
            body = response.json() if status == 200 and raw else []
            error = None
        except requests.HTTPError as exc:
            # 204 (empty activity) is normally returned without an error by requests; retain any other status.
            response = getattr(exc, "response", None)
            status = response.status_code if response is not None else None
            raw = response.content if response is not None else b""
            body = []
            error = str(exc)
        except requests.RequestException as exc:
            status, raw, body, error = None, b"", [], str(exc)
        safe = repo["full_name"].replace("/", "__")
        saved = save_bytes(destination / "github" / f"{safe}.json", raw)
        records.append({"full_name": repo["full_name"], "status_code": status,
                        "fetched_at_utc": utc_now(), "response_sha256": saved["sha256"],
                        "response_headers": dict(response.headers) if response is not None else {},
                        "snapshot_path": saved["path"], "error": error,
                        "weekly_activity": body if status == 200 else []})
    return {"source": "GitHub REST stats/commit_activity", "repositories": records}


def freeze_study(*, force: bool = False) -> Path:
    if MANIFEST.exists() and not force:
        raise FileExistsError(f"Frozen manifest already exists: {MANIFEST}; use --force only to replace it")
    study = read_study()
    frozen_at = datetime.now(ZoneInfo("America/Toronto"))
    freeze_id = "freeze_" + frozen_at.strftime("%Y%m%dT%H%M%S%z")
    destination = SNAPSHOTS / freeze_id
    destination.mkdir(parents=True, exist_ok=False)
    cohort = discover_repositories(destination)
    write_json(destination / "github_cohort.json", cohort)
    repos = [repo for language in study["domains"]["github"]["languages"] for repo in cohort[language]]
    github = download_github_activity(destination, repos)
    write_json(destination / "github_activity_manifest.json", github)
    ieso = download_ieso_snapshot(destination, through_year=frozen_at.year)
    write_json(destination / "ieso_manifest.json", ieso)
    revisions = resolve_model_revisions()
    write_json(destination / "model_revisions.json", revisions)
    write_json(MANIFEST, {
        "study_id": study["study_id"], "protocol_version": study["protocol_version"],
        "frozen_at_local": frozen_at.isoformat(), "freeze_snapshot_id": freeze_id,
        "repository_cohort": cohort, "model_revisions": revisions,
        "source_manifests": {"ieso": ieso, "github": github},
        "cutoffs": [d.isoformat() for d in second_tuesdays(
            datetime.fromisoformat(str(study["prospective"]["start_date"])).date(),
            int(study["prospective"]["count"]))],
    })
    return destination


def repositories_from_manifest() -> list[dict[str, Any]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cohort = manifest["repository_cohort"]
    study = read_study()
    return [repo for language in study["domains"]["github"]["languages"] for repo in cohort[language]]


def collect_source_snapshot(snapshot_id: str, *, destination: Path | None = None) -> tuple[Path, dict[str, Any]]:
    if not MANIFEST.exists():
        raise FileNotFoundError("Run `freeze` before collecting cutoff data")
    destination = destination or SNAPSHOTS / snapshot_id
    if destination.exists():
        raise FileExistsError(f"Snapshot already exists; snapshots are immutable: {destination}")
    destination.mkdir(parents=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ieso = download_ieso_snapshot(destination)
    github = download_github_activity(destination, repositories_from_manifest())
    combined = {"snapshot_id": snapshot_id, "created_at_utc": utc_now(), "ieso": ieso, "github": github}
    write_json(destination / "snapshot_manifest.json", combined)
    return destination, combined

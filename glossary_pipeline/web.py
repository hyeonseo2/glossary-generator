import argparse
import html
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from typing import List, Tuple

from .config import PipelineConfig
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "samples"
EN_DIR = SAMPLES_DIR / "en"
KO_DIR = SAMPLES_DIR / "ko"
OUT_DIR = ROOT / "out"
WEB_OUT_DIR = OUT_DIR / "web"

RUN_JOBS = {}
_EMBEDDER = None


def _get_embedder(model_name: str):
    global _EMBEDDER
    if _EMBEDDER is not None and getattr(_EMBEDDER, "__dict__", {}).get("__model_name__") == model_name:
        return _EMBEDDER
    obj = SentenceTransformer(model_name)
    setattr(obj, "__model_name__", model_name)
    _EMBEDDER = obj
    return obj

def now_ts():
    return int(time.time())


def discover_pairs() -> List[Tuple[str, str, str]]:
    """Discover paired documents by filename stem in samples/en and samples/ko."""
    if not EN_DIR.exists() or not KO_DIR.exists():
        return []

    allowed = {".md", ".txt", ".markdown", ".html"}
    en_files = {
        p.stem: str(p)
        for p in EN_DIR.glob("*")
        if p.is_file() and p.suffix.lower() in allowed
    }

    pairs = []
    for ko_path in KO_DIR.glob("*"):
        if not ko_path.is_file() or ko_path.suffix.lower() not in allowed:
            continue
        if ko_path.stem in en_files:
            pairs.append((ko_path.stem, en_files[ko_path.stem], str(ko_path)))

    return sorted(pairs, key=lambda p: p[0].lower())


def _start_job() -> str:
    job_id = f"job-{uuid.uuid4().hex[:8]}-{int(time.time())}"
    RUN_JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "step": "대기 중",
        "message": "작업이 등록되었습니다.",
        "updated_at": now_ts(),
        "result": None,
        "error": None,
        "out_json": None,
        "out_csv": None,
    }
    return job_id


def _safe_set(job_id: str, **kwargs):
    if job_id not in RUN_JOBS:
        return
    RUN_JOBS[job_id].update(kwargs)
    RUN_JOBS[job_id]["updated_at"] = now_ts()


def _get_active_job():
    """Return latest queued/running job first, otherwise latest updated job."""
    if not RUN_JOBS:
        return None
    active = {jid: info for jid, info in RUN_JOBS.items() if info.get("status") in ("queued", "running")}
    if active:
        return max(active.items(), key=lambda kv: kv[1].get("updated_at", 0))
    return max(RUN_JOBS.items(), key=lambda kv: kv[1].get("updated_at", 0))


def _run_job(job_id: str, out_json: str, out_csv: str, *, topk: int = 80, strict_filter: bool = True, min_score_mean: float = 0.74, min_final_confidence: float = 0.8, max_ko_ngram: int = 3):
    from .pipeline import GlossaryPipeline

    try:
        _safe_set(job_id, status="running", progress=2, step="문서 매칭", message="samples/en, samples/ko 쌍 확인")
        pairs = discover_pairs()
        if not pairs:
            raise RuntimeError("No matching EN/KO document pairs found")

        _safe_set(job_id, progress=8, step="문서 로드", message=f"{len(pairs)}개 문서 쌍 로드")
        docs = []
        total = len(pairs)
        for i, (source_id, en_path, ko_path) in enumerate(pairs, start=1):
            with open(en_path, "r", encoding="utf-8") as f:
                en_text = f.read()
            with open(ko_path, "r", encoding="utf-8") as f:
                ko_text = f.read()
            docs.append((source_id, en_text, ko_text))
            _safe_set(job_id, progress=8 + int(i / max(total, 1) * 30), message=f"문서 로드 {i}/{total}")

        _safe_set(job_id, progress=40, step="파이프라인 실행", message="추출/정렬/채점 진행")
        cfg = PipelineConfig()
        cfg.strict_filter = strict_filter
        cfg.min_score_mean = min_score_mean
        cfg.min_final_confidence = min_final_confidence
        cfg.topk = topk
        cfg.max_ko_ngram = max_ko_ngram
        pipe = GlossaryPipeline(cfg, embed_model=_get_embedder(cfg.embedding_model_name))

        def progress_cb(stage: str, i: int, total: int, source_id: str):
            if stage == "run_pair_start":
                pct = 40 + int((i - 1) / max(1, total) * 45)
                _safe_set(
                    job_id,
                    progress=min(84, pct),
                    step="파이프라인 실행",
                    message=f"{source_id}: 문서 처리 중 ({i}/{total})",
                )
            elif stage == "run_pair_done":
                pct = 40 + int(i / max(1, total) * 45)
                _safe_set(
                    job_id,
                    progress=min(85, pct),
                    step="파이프라인 실행",
                    message=f"{source_id}: 처리 완료 ({i}/{total})",
                )
            elif stage == "aggregate":
                _safe_set(job_id, progress=86, step="집계", message="집계/후처리 중")

        rows = pipe.fit_transform(docs, progress_cb=progress_cb)

        _safe_set(job_id, progress=87, step="집계", message="집계 완료, 저장 준비")

        _safe_set(job_id, progress=85, step="결과 저장", message=f"{len(rows)}개 항목 저장")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        WEB_OUT_DIR.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        marker = job_id.split("-")[1] if "-" in job_id else "run"
        backup_base = f"all_glossary_run_{ts}_{marker}"

        # Save every run with a unique timestamped filename.
        target_json = str(WEB_OUT_DIR / f"{backup_base}.json")
        target_csv = str(WEB_OUT_DIR / f"{backup_base}.csv")
        pipe.export(rows, out_json=target_json, out_csv=target_csv)

        # Keep canonical files as latest good results only (when non-empty).
        canonical_json = out_json
        canonical_csv = out_csv
        latest_json_name = Path(target_json).name
        latest_csv_name = Path(target_csv).name
        if len(rows) > 0:
            # overwrite canonical with best-effort latest good run
            import shutil

            Path(canonical_json).write_text(Path(target_json).read_text(encoding="utf-8"), encoding="utf-8")
            Path(canonical_csv).write_text(Path(target_csv).read_text(encoding="utf-8"), encoding="utf-8")
            latest_json_name = Path(canonical_json).name
            latest_csv_name = Path(canonical_csv).name
        else:
            # if empty run, do not replace canonical output
            latest_json_name = Path(canonical_json).name if Path(canonical_json).exists() else Path(target_json).name
            latest_csv_name = Path(canonical_csv).name if Path(canonical_csv).exists() else Path(target_csv).name

        _safe_set(
            job_id,
            status="completed",
            progress=100,
            step="완료",
            message=f"완료: {len(rows)}개 항목",
            result={
                "entries": len(rows),
                "source_count": len(docs),
                "out_json": latest_json_name,
                "out_csv": latest_csv_name,
                "canonical_out_json": Path(canonical_json).name if len(rows) > 0 else Path(out_json).name,
            },
            out_json=latest_json_name,
            out_csv=latest_csv_name,
        )
    except Exception as e:
        _safe_set(job_id, status="failed", progress=100, step="실패", message=f"오류: {e}", error=str(e))


def run_all_documents_async():
    pass


def run_all_documents(out_json: str = None, out_csv: str = None, cfg_path: str = None, strict_filter: bool = False, min_score_mean: float = 0.62, min_final_confidence: float = 0.70):
    from .pipeline import GlossaryPipeline

    pairs = discover_pairs()
    if not pairs:
        raise RuntimeError("No matching EN/KO document pairs found")

    docs = []
    for source_id, en_path, ko_path in pairs:
        with open(en_path, "r", encoding="utf-8") as f:
            en_text = f.read()
        with open(ko_path, "r", encoding="utf-8") as f:
            ko_text = f.read()
        docs.append((source_id, en_text, ko_text))

    cfg = PipelineConfig()
    if cfg_path:
        with open(cfg_path, "r", encoding="utf-8") as f:
            over = json.load(f)
        for k, v in over.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    # Prefer stricter defaults for better quality in web batch runs
    cfg.strict_filter = bool(strict_filter)
    cfg.min_score_mean = min_score_mean
    cfg.min_final_confidence = min_final_confidence

    pipe = GlossaryPipeline(cfg, embed_model=_get_embedder(cfg.embedding_model_name))
    rows = pipe.fit_transform(docs)

    out_json = out_json or (WEB_OUT_DIR / "all_glossary.json").as_posix()
    out_csv = out_csv or (WEB_OUT_DIR / "all_glossary.csv").as_posix()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WEB_OUT_DIR.mkdir(parents=True, exist_ok=True)

    pipe.export(rows, out_json=out_json, out_csv=out_csv)
    return rows, out_json, out_csv, len(docs)


class WebHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, content_type: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _ui_text(self, lang: str = "en"):
        en = {
            "title": "Translation Glossary Generator",
            "subtitle": "Extract terms from matched documents in English/Korean folders.",
            "doc_label": "1) Source documents",
            "path_label": "English file path: samples/en, Korean file path: samples/ko",
            "matched": "Matched document pairs",
            "columns": ["No.", "Document ID", "English file", "Korean file"],
            "run_btn": "Run term extraction",
            "running_card": "Current progress",
            "state_idle": "idle / 0%",
            "result_title": "2) Results",
            "result_file": "Result file",
            "result_job": "Action",
            "preview_title": "Latest glossary preview",
            "table_headers": ["#", "English term", "Korean term", "Confidence", "Count", "Example (EN)", "Example (KO)"],
            "no_pairs": "No matched documents.",
            "no_results": "No data available.",
            "preview_link": "View full table",
            "no_preview": "No preview available yet.",
            "run_in_progress": "Starting...",
            "lang_switch": "Language",
            "lang_en": "English",
            "lang_ko": "Korean",
        }
        ko = {
            "title": "번역 용어집 생성기",
            "subtitle": "영문/국문 폴더의 같은 이름 문서를 기준으로 용어를 추출합니다.",
            "doc_label": "1) 대상 문서",
            "path_label": "영문 파일 경로: samples/en, 국문 파일 경로: samples/ko",
            "matched": "매칭된 문서 수",
            "columns": ["No.", "문서 ID", "영문 파일", "국문 파일"],
            "run_btn": "용어 추출 실행",
            "running_card": "현재 작업 진행",
            "state_idle": "idle / 0%",
            "result_title": "2) 생성된 결과",
            "result_file": "결과 파일",
            "result_job": "작업",
            "preview_title": "최근 용어집 미리보기",
            "table_headers": ["#", "영문 용어", "한글 용어", "신뢰도", "근거 수", "예시(영문)", "예시(한글)"],
            "no_pairs": "매칭된 문서가 없습니다.",
            "no_results": "표시할 데이터가 없습니다.",
            "preview_link": "전체 표로 보기",
            "no_preview": "아직 미리볼 결과가 없습니다.",
            "run_in_progress": "시작 중...",
            "lang_switch": "언어",
            "lang_en": "English",
            "lang_ko": "한국어",
        }
        return ko if lang == "ko" else en

    def _pairs_table_html(self, lang: str = "en"):
        T = self._ui_text(lang)
        rows = discover_pairs()
        if not rows:
            return f"<tr><td colspan=4>{T['no_pairs']}</td></tr>"
        out = []
        for i, (sid, en_path, ko_path) in enumerate(rows, 1):
            out.append(
                f"<tr><td>{i}</td><td>{html.escape(sid)}</td><td>{html.escape(Path(en_path).name)}</td><td>{html.escape(Path(ko_path).name)}</td></tr>"
            )
        return "\\n".join(out)

    def _load_json_rows(self, name: str):
        target = WEB_OUT_DIR / name
        if not target.exists() or not target.is_file():
            return None
        try:
            rows = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            return None
        return rows if isinstance(rows, list) else None

    def _recent_outputs_html(self):
        WEB_OUT_DIR.mkdir(parents=True, exist_ok=True)
        rows = sorted(WEB_OUT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not rows:
            return "<tr><td colspan=3>No outputs yet.</td></tr>", None

        html_rows = []
        latest_nonempty = None
        canonical = WEB_OUT_DIR / "all_glossary.json"

        for pth in rows[:30]:
            parsed = self._load_json_rows(pth.name)
            has_rows = bool(parsed)
            if latest_nonempty is None and has_rows:
                latest_nonempty = pth

            name = html.escape(pth.name)
            csv = name.replace(".json", ".csv")
            csv_link = f'/output/{html.escape(csv)}' if (WEB_OUT_DIR / csv).exists() else "CSV"
            csv_a = f'<a class="link" href="{csv_link}">CSV</a>' if csv_link != "CSV" else "CSV"
            html_rows.append(
                f"<tr><td>{name}</td>"
                f"<td>"
                f"  <a class='link' href='/output/{name}'>JSON</a>"
                f"  <a class='link' href='/output/{name}?view=table'>Table</a>"
                f"  {csv_a}"
                f"</td></tr>"
            )

        canonical_rows = self._load_json_rows(canonical.name)
        latest = canonical if canonical_rows is not None else latest_nonempty
        if latest is None:
            return "".join(html_rows), None

        return "".join(html_rows), latest.name

    def _truncate_preview(self, text: str, limit: int = 140):
        if text is None:
            text = ""
        s = str(text)
        if len(s) <= limit:
            return html.escape(s), "", False
        cut = limit
        space_pos = s.rfind(" ", 0, cut)
        if space_pos > 0:
            cut = space_pos
        return html.escape(s[:cut]), html.escape(s[cut:]), True

    def _json_table_rows(self, rows, lang: str = "en") -> str:
        T = self._ui_text(lang)
        if not isinstance(rows, list) or not rows:
            return f'<tr><td colspan="7">{T["no_results"]}</td></tr>'
        out = []
        for i, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                continue
            en_term = html.escape(str(row.get("en_term", "")))
            ko_term = html.escape(str(row.get("ko_term", "")))
            conf = html.escape(str(row.get("confidence", "")))
            count = html.escape(str(row.get("evidence_count", "")))

            ex_en_short, ex_en_rest, ex_en_trunc = self._truncate_preview(str(row.get("example_en", "")), limit=120)
            ex_ko_short, ex_ko_rest, ex_ko_trunc = self._truncate_preview(str(row.get("example_ko", "")), limit=120)

            ex_en_ellipsis = '<span class="term-ellipsis">...</span>' if ex_en_trunc else ''
            ex_ko_ellipsis = '<span class="term-ellipsis">...</span>' if ex_ko_trunc else ''

            ex_en_cell = (
                f'<span class="term-expand" onclick="this.classList.toggle(\'open\')">'
                f'<span class="term-preview">{ex_en_short}</span>'
                f"{ex_en_ellipsis}"
                f'<span class="term-rest">{ex_en_rest}</span>'
                f"</span>"
            )
            ex_ko_cell = (
                f'<span class="term-expand" onclick="this.classList.toggle(\'open\')">'
                f'<span class="term-preview">{ex_ko_short}</span>'
                f"{ex_ko_ellipsis}"
                f'<span class="term-rest">{ex_ko_rest}</span>'
                f"</span>"
            )

            out.append(
                f'<tr><td>{i}</td><td>{en_term}</td><td>{ko_term}</td><td>{conf}</td>'
                f'<td>{count}</td><td>{ex_en_cell}</td><td>{ex_ko_cell}</td></tr>'
            )

        if not out:
            return f'<tr><td colspan="7">{T["no_results"]}</td></tr>'
        return "".join(out)

    def _inline_table_html(self, name: str, lang: str = "en") -> str:
        T = self._ui_text(lang)
        rows = self._load_json_rows(name)
        if rows is None:
            rows = self._load_json_rows("all_glossary.json")
        if rows is None:
            return f"<p>{T['no_preview']}</p>"

        head = "".join(f"<th>{h}</th>" for h in T["table_headers"])
        body = self._json_table_rows(rows, lang=lang)
        return f"""
      <div style="overflow:auto; margin-top:8px; margin-bottom:8px;">
        <table>
          <thead><tr>{head}</tr></thead>
          <tbody>{body}</tbody>
        </table>
      </div>
      <p><a class="link" href="/output/{html.escape(name)}?view=table&lang={lang}">{T['preview_link']}</a></p>
      """

    def _json_table_html(self, name: str, lang: str = "en") -> str:
        T = self._ui_text(lang)
        target = WEB_OUT_DIR / name
        try:
            rows = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            return "<h3>JSON parse error</h3><p>Unable to render table.</p>"
        if not isinstance(rows, list):
            return "<h3>Unsupported format</h3><p>Expected a term-array JSON.</p>"

        head = "".join(f"<th>{h}</th>" for h in T["table_headers"])
        body = self._json_table_rows(rows, lang=lang)
        return (
            '<html><head><meta charset="utf-8" /><title>Glossary table</title>'
            '<style> body{margin:0;padding:16px;font-family:Arial,sans-serif} '
            'table{width:100%;border-collapse:collapse;min-width:1000px;} '
            'th,td{border:1px solid #e2e8f0;padding:8px;white-space:nowrap; vertical-align: top;} '
            '.term-expand{cursor:pointer; display:inline; max-width:420px; white-space:pre-wrap; overflow-wrap:anywhere;} '
            '.term-preview{display:inline; color:#111827;} '
            '.term-rest{display:none;} '
            '.term-ellipsis{display:inline;} '
            '.term-expand.open .term-rest{display:inline;} '
            '.term-expand.open .term-ellipsis{display:none;} '
            '</style></head><body>'
            f'<h3>{T["preview_title"]} - {html.escape(name)}</h3>'
            f'<div style="overflow:auto;"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
            f'<p><a href="/?lang={lang}">Home</a> | <a href="/output/{html.escape(name)}">JSON</a></p>'
            '</body></html>'
        )

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qlang = parse_qs(parsed.query).get("lang", [""])[0]
        lang = qlang if qlang in ("ko", "en") else "en"
        T = self._ui_text(lang)

        if path == "/":
            pairs_len = len(discover_pairs())
            pairs_html = self._pairs_table_html(lang)
            outputs_html, latest_json = self._recent_outputs_html()
            web_path = html.escape(str(WEB_OUT_DIR))
            preview_html = self._inline_table_html(latest_json, lang) if latest_json else f"<p>{T['no_preview']}</p>"

            q = parse_qs(parsed.query)
            initial_job_id = (q.get("job_id") or [""])[0]
            if not initial_job_id:
                active = _get_active_job()
                if active:
                    initial_job_id = active[0]
            html_body = f'''<!doctype html>
<html><head><meta charset="utf-8"/><title>{T["title"]}</title>
<style>
:root {{--bg:#f8fafc;--card:#fff;--line:#e2e8f0;--text:#0f172a;--muted:#64748b;--primary:#2563eb;--primary-dark:#1d4ed8;}}
* {{box-sizing:border-box;}}
body{{margin:0;padding:24px;font-family:Inter,Arial,sans-serif;background:var(--bg);color:var(--text);line-height:1.45;}}
.page{{max-width:1200px;margin:0 auto;}}
h2,h3{{margin:0 0 8px 0;}}
.muted{{color:var(--muted);font-size:14px;}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin-top:18px;box-shadow:0 6px 16px rgba(15,23,42,.05);overflow:auto;}}
.toolbar{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:8px 0 14px;}}
.btn{{border:0;color:#fff;background:var(--primary);padding:10px 16px;border-radius:10px;font-weight:700;cursor:pointer;text-decoration:none;display:inline-block;}}
.btn:hover{{background:var(--primary-dark);}}
table{{width:100%;border-collapse:collapse;font-size:14px;min-width:900px;}}
thead th,th,td{{border:1px solid var(--line);padding:10px;text-align:left;vertical-align:top;white-space:nowrap;}}
.term-expand{{cursor:pointer;display:inline;max-width:420px;white-space:pre-wrap;overflow-wrap:anywhere;}}
.term-preview{{display:inline;color:#111827;}}
.term-rest{{display:none;}}
.term-ellipsis{{display:inline;}}
.term-expand.open .term-rest{{display:inline;}}
.term-expand.open .term-ellipsis{{display:none;}}
thead th{{position:sticky;top:0;z-index:1;background:#f1f5f9;border-bottom:2px solid #94a3b8;}}
tbody tr:nth-child(odd){{background:#f8fafc;}}tbody tr:hover{{background:#eff6ff;}}
.link{{color:var(--primary);text-decoration:none;}}
</style>
</head><body><div class="page">
<h2>{T["title"]}</h2>
<p class="muted">{T["subtitle"]}</p>
<div class="toolbar">
  <span class="muted">{T["lang_switch"]}:</span>
  <a class="btn" href="/?lang=en">{T["lang_en"]}</a>
  <a class="btn" href="/?lang=ko">{T["lang_ko"]}</a>
</div>
<div class="card"><h3>{T["doc_label"]}</h3><p class="muted">{T["path_label"]}</p>
<div class="toolbar"><span class="muted">{T["matched"]}: {pairs_len}</span></div>
<div style="overflow:auto;"><table><thead><tr><th>{T["columns"][0]}</th><th>{T["columns"][1]}</th><th>{T["columns"][2]}</th><th>{T["columns"][3]}</th></tr></thead><tbody>{pairs_html}</tbody></table></div></div>
<div class="toolbar">
  <button class="btn" id="run-all-btn" type="button" data-mode="normal">{T["run_btn"]}</button>
</div>
<div class="card" id="run-progress-card" style="display:none">
  <h3>{T["running_card"]}</h3>
  <p><span id="home-state">{T["state_idle"]}</span></p>
  <div class="bar-wrap" style="height:14px;background:#e2e8f0;border-radius:8px;overflow:hidden"><div id="home-bar" class="bar" style="height:14px;background:#2563eb;width:0%;"></div></div>
  <p id="home-step"></p>
</div>
<div class="card"><h3>{T["result_title"]}</h3><div style="overflow:auto;"><table><thead><tr><th>{T["result_file"]}</th><th>{T["result_job"]}</th></tr></thead><tbody>{outputs_html}</tbody></table></div></div>
<div class="card"><h3>{T["preview_title"]}</h3>{preview_html}</div>
</div>
<script>
(function() {{
  const q = new URLSearchParams(window.location.search);
  const jobId = q.get('job_id') || '';
  const currentLang = q.get('lang') === 'ko' ? 'ko' : 'en';
  const stateEl = document.getElementById('home-state');
  const stepEl = document.getElementById('home-step');
  const barEl = document.getElementById('home-bar');
  const cardEl = document.getElementById('run-progress-card');
  const runBtn = document.getElementById('run-all-btn');

  function toEn(text) {{
    const s = String(text || '');
    const map = {{
      '실패': 'failed',
      '완료': 'completed',
      '문서 매칭': 'Document matching',
      '문서 로드': 'Loading documents',
      '파이프라인 실행': 'Pipeline execution',
      '집계': 'Aggregation',
      '결과 저장': 'Saving results',
      '추출/정렬/채점 진행': 'Extract/align/score in progress',
      '집계/후처리 중': 'Aggregating/post-processing',
      '집계 완료, 저장 준비': 'Aggregation complete, preparing save',
      '처리 완료': 'completed',
      '문서 처리 중': 'processing document',
      '개 항목 저장': ' entries saved',
      '개 문서 쌍 로드': ' document pairs loaded',
      '오류:': 'Error:',
    }};
    let out = s;
    Object.keys(map).forEach((k) => {{
      out = out.split(k).join(map[k]);
    }});
    return out;
  }}

  function localizeStatus(s) {{
    const raw = String(s || '');
    if (currentLang === 'ko') return raw;
    const map = {{ queued: 'queued', running: 'running', completed: 'completed', failed: 'failed' }};
    return map[raw] || toEn(raw);
  }}

  function localizeStep(s) {{
    if (currentLang === 'ko') return String(s || '');
    return toEn(s || '');
  }}

  function localizeMessage(s) {{
    if (currentLang === 'ko') return String(s || '');
    return toEn(s || '');
  }}

  async function poll(jid) {{
    if (!jid) return;
    try {{
      const r = await fetch('/api/run-status?job_id=' + encodeURIComponent(jid));
      if (!r.ok) return;
      const d = await r.json();
      cardEl.style.display = 'block';
      stateEl.textContent = localizeStatus(d.status) + ' / ' + (d.progress || 0) + '%';
      stepEl.textContent = localizeStep(d.step) + ' - ' + localizeMessage(d.message);
      barEl.style.width = String(d.progress || 0) + '%';
      if (d.status === 'running' || d.status === 'queued') {{
        setTimeout(() => poll(jid), 1200);
      }} else if (d.status === 'completed' || d.status === 'failed') {{
        setTimeout(() => {{ location.reload(); }}, 1000);
      }}
    }} catch (e) {{
      setTimeout(() => poll(jid), 1800);
    }}
  }}

  const startRun = async (endpoint, btnEl) => {{
    btnEl.disabled = true;
    const old = btnEl.textContent;
    btnEl.textContent = currentLang === 'ko' ? '시작 중...' : 'Starting...';
    const r = await fetch(endpoint, {{ method: 'POST', headers: {{ 'Accept': 'application/json' }} }});
    btnEl.disabled = false;
    btnEl.textContent = old;
    if (!r.ok) return;
    const d = await r.json();
    const nextId = d && d.job_id ? d.job_id : '';
    if (!nextId) {{
      location.reload();
      return;
    }}
    const next = new URL(window.location.href);
    next.searchParams.set('job_id', nextId);
    history.replaceState(null, '', next.toString());
    poll(nextId);
  }};

  if (runBtn) {{
    runBtn.addEventListener('click', async () => {{
      await startRun('/run-all?lang=' + currentLang, runBtn);
    }});
  }}

  if (jobId) {{
    poll(jobId);
  }}

}})();
</script>
</body></html>'''
            self._send(200, "text/html; charset=utf-8", html_body.encode("utf-8"))
            return

        if path == "/api/run-status":
            q = parse_qs(parsed.query)
            job_id = (q.get("job_id") or [""])[0]
            info = RUN_JOBS.get(job_id)
            if not info:
                self._send(404, "application/json; charset=utf-8", b'{"error":"not_found"}')
                return
            payload = dict(info)
            payload["job_id"] = job_id
            self._send(200, "application/json; charset=utf-8", json.dumps(payload).encode("utf-8"))
            return

        if path == "/run-all":
            html_body = """<!doctype html><html><head><meta charset='utf-8'><title>run-all</title>
<style>body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc}.card{max-width:640px;margin:30px auto;background:#fff;border:1px solid #e2e8f0;padding:16px;border-radius:10px}.btn{border:0;background:#2563eb;color:#fff;padding:10px 12px;border-radius:8px;text-decoration:none}</style>
</head><body><div class='card'><h3>/run-all is POST only</h3><p>Use the home button or submit below.</p><form method='post' action='/run-all'><button class='btn' type='submit'>Run</button></form></div></body></html>"""
            self._send(200, "text/html; charset=utf-8", html_body.encode("utf-8"))
            return

        if path.startswith("/output/"):
            name = path.split("/", 2)[2]
            target = WEB_OUT_DIR / name
            if not target.exists() or not target.is_file():
                self._send(404, "text/plain; charset=utf-8", b"Not Found")
                return
            view = (parse_qs(parsed.query).get("view") or [""])[0]
            if target.suffix == ".json" and view == "table":
                view_lang = (parse_qs(parsed.query).get("lang") or [lang])[0]
                view_lang = view_lang if view_lang in ("ko", "en") else lang
                self._send(200, "text/html; charset=utf-8", self._json_table_html(name, view_lang).encode("utf-8"))
                return
            body = target.read_bytes()
            ctype = "application/json; charset=utf-8" if target.suffix == ".json" else "text/csv; charset=utf-8"
            self._send(200, ctype, body)
            return

        if path == "/api/pairs":
            rows = [{"source_id": sid, "en_path": en_path, "ko_path": ko_path} for sid, en_path, ko_path in discover_pairs()]
            self._send(200, "application/json; charset=utf-8", json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8"))
            return

        self._send(404, "text/plain; charset=utf-8", b"Not Found")

    def do_HEAD(self):
        # Explicitly support HEAD for compatibility with clients/probers that use it.
        if self.path == "/api/run-status":
            # Keep auth-likely lightweight status ping
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if self.path.startswith("/output/") or self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", "0")
        self.end_headers()


    def do_POST(self):
        parsed_path = urlparse(self.path)
        post_lang = (parse_qs(parsed_path.query).get("lang") or ["en"])[0]
        post_lang = post_lang if post_lang in ("ko", "en") else "en"
        if parsed_path.path != "/run-all":
            self._send(404, "text/plain; charset=utf-8", b"Not Found")
            return

        # allow one running job max
        if any(v.get("status") in ("running", "queued") for v in RUN_JOBS.values()):
            self._send(409, "text/plain; charset=utf-8", "이미 실행 중인 작업이 있습니다.".encode("utf-8"))
            return

        job_id = _start_job()
        out_json = str(WEB_OUT_DIR / "all_glossary.json")
        out_csv = str(WEB_OUT_DIR / "all_glossary.csv")
        # 정밀 모드만 사용
        topk = 2000
        min_score = 0.62
        min_conf = 0.70
        max_ko_ngram = 4

        threading.Thread(
            target=_run_job,
            args=(job_id, out_json, out_csv),
            kwargs={
                "topk": topk,
                "strict_filter": True,
                "min_score_mean": min_score,
                "min_final_confidence": min_conf,
                "max_ko_ngram": max_ko_ngram,
            },
            daemon=True,
        ).start()

        accept = self.headers.get("Accept", "")
        if "application/json" in accept:
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            payload = {"job_id": job_id}
            body = json.dumps(payload)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            return

        loc = f"/?job_id={job_id}&lang={post_lang}"
        self.send_response(303)
        self.send_header("Location", loc)
        self.end_headers()


def run_server(host: str = "127.0.0.1", port: int = 7860):
    server = HTTPServer((host, port), WebHandler)
    print(f"EN-KO glossary web manager running at http://{host}:{port}")
    server.serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

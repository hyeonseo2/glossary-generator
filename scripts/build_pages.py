import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "out" / "web" / "all_glossary.json"
OUT_CSV = ROOT / "out" / "web" / "all_glossary.csv"
SAMPLES_EN = ROOT / "samples" / "en"
SAMPLES_KO = ROOT / "samples" / "ko"
SITE = ROOT / "site"
DATA_DIR = SITE / "data"


def esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def discover_pairs():
    if not SAMPLES_EN.exists() or not SAMPLES_KO.exists():
        return []
    ko_map = {p.stem: p for p in SAMPLES_KO.iterdir() if p.is_file()}
    rows = []
    for ep in sorted([p for p in SAMPLES_EN.iterdir() if p.is_file()], key=lambda x: x.name):
        kp = ko_map.get(ep.stem)
        if kp:
            rows.append((ep.stem, ep, kp))
    return rows


def build_rows_html(rows):
    out = []
    for i, r in enumerate(rows, 1):
        en_term = esc(r.get("en_term", ""))
        ko_term = esc(r.get("ko_term", ""))
        conf = esc(r.get("confidence", ""))
        count = esc(r.get("evidence_count", ""))

        ex_en = str(r.get("example_en", ""))
        ex_ko = str(r.get("example_ko", ""))

        ex_en_preview = esc(ex_en[:120])
        ex_ko_preview = esc(ex_ko[:120])
        ex_en_rest = esc(ex_en[120:])
        ex_ko_rest = esc(ex_ko[120:])
        ex_en_ellipsis = "<span class='term-ellipsis'>...</span>" if len(ex_en) > 120 else ""
        ex_ko_ellipsis = "<span class='term-ellipsis'>...</span>" if len(ex_ko) > 120 else ""

        ex_en_html = (
            f"<span class='term-expand' onclick=\"this.classList.toggle('open')\">"
            f"<span class='term-preview'>{ex_en_preview}</span>{ex_en_ellipsis}<span class='term-rest'>{ex_en_rest}</span></span>"
        )
        ex_ko_html = (
            f"<span class='term-expand' onclick=\"this.classList.toggle('open')\">"
            f"<span class='term-preview'>{ex_ko_preview}</span>{ex_ko_ellipsis}<span class='term-rest'>{ex_ko_rest}</span></span>"
        )

        out.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{en_term}</td>"
            f"<td>{ko_term}</td>"
            f"<td>{conf}</td>"
            f"<td>{count}</td>"
            f"<td>{ex_en_html}</td>"
            f"<td>{ex_ko_html}</td>"
            "</tr>"
        )
    return "".join(out)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    if OUT_JSON.exists():
        try:
            obj = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                rows = obj
        except Exception:
            rows = []

    (DATA_DIR / "latest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if OUT_CSV.exists():
        (DATA_DIR / "latest.csv").write_bytes(OUT_CSV.read_bytes())

    pairs = discover_pairs()
    pair_rows = []
    for i, (sid, enp, kop) in enumerate(pairs, 1):
        pair_rows.append(
            f"<tr><td>{i}</td><td>{esc(sid)}</td><td>{esc(enp.name)}</td><td>{esc(kop.name)}</td></tr>"
        )

    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    glossary_rows_html = build_rows_html(rows)

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\"/>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
  <title>Translation Glossary Generator</title>
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
    thead th{{position:sticky;top:0;z-index:1;background:#f1f5f9;border-bottom:2px solid #94a3b8;}}
    tbody tr:nth-child(odd){{background:#f8fafc;}}tbody tr:hover{{background:#eff6ff;}}
    .link{{color:var(--primary);text-decoration:none;}}

    .term-expand{{cursor:pointer;display:inline;max-width:420px;white-space:pre-wrap;overflow-wrap:anywhere;}}
    .term-preview{{display:inline;color:#111827;}}
    .term-rest{{display:none;}}
    .term-ellipsis{{display:inline;}}
    .term-expand.open .term-rest{{display:inline;}}
    .term-expand.open .term-ellipsis{{display:none;}}
  </style>
</head>
<body>
  <div class=\"page\">
    <h2 id=\"title\">Translation Glossary Generator</h2>
    <p class=\"muted\" id=\"subtitle\">Extract terms from matched documents in English/Korean folders.</p>

    <div class=\"toolbar\">
      <span class=\"muted\" id=\"langLabel\">Language:</span>
      <button class=\"btn\" onclick=\"setLang('en')\">English</button>
      <button class=\"btn\" onclick=\"setLang('ko')\">한국어</button>
      <a class=\"btn\" href=\"https://github.com/hyeonseo2/glossary-generator/actions/workflows/pages.yml\" target=\"_blank\" rel=\"noreferrer\" id=\"runBtn\">Run term extraction</a>
    </div>

    <div class=\"card\">
      <h3 id=\"docLabel\">1) Source documents</h3>
      <p class=\"muted\" id=\"pathLabel\">English file path: samples/en, Korean file path: samples/ko</p>
      <div class=\"toolbar\"><span class=\"muted\" id=\"matchedLabel\">Matched document pairs: {len(pairs)}</span></div>
      <div style=\"overflow:auto;\">
        <table>
          <thead><tr><th>No.</th><th id=\"colDoc\">Document ID</th><th id=\"colEnFile\">English file</th><th id=\"colKoFile\">Korean file</th></tr></thead>
          <tbody>{''.join(pair_rows) if pair_rows else '<tr><td colspan="4">No matched documents.</td></tr>'}</tbody>
        </table>
      </div>
    </div>

    <div class=\"card\">
      <h3 id=\"resultTitle\">2) Results</h3>
      <p class=\"muted\" id=\"updated\">Updated: {updated}</p>
      <div class=\"toolbar\">
        <a class=\"btn\" href=\"./data/latest.json\">JSON</a>
        <a class=\"btn\" href=\"./data/latest.csv\">CSV</a>
      </div>
      <div style=\"overflow:auto;\">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th id=\"thEnTerm\">English term</th>
              <th id=\"thKoTerm\">Korean term</th>
              <th id=\"thConf\">Confidence</th>
              <th id=\"thCount\">Count</th>
              <th id=\"thExEn\">Example (EN)</th>
              <th id=\"thExKo\">Example (KO)</th>
            </tr>
          </thead>
          <tbody>
            {glossary_rows_html if glossary_rows_html else '<tr><td colspan="7">No data available.</td></tr>'}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const I18N = {{
      en: {{
        title: 'Translation Glossary Generator',
        subtitle: 'Extract terms from matched documents in English/Korean folders.',
        langLabel: 'Language:',
        runBtn: 'Run term extraction',
        docLabel: '1) Source documents',
        pathLabel: 'English file path: samples/en, Korean file path: samples/ko',
        matchedLabel: 'Matched document pairs: {len(pairs)}',
        colDoc: 'Document ID',
        colEnFile: 'English file',
        colKoFile: 'Korean file',
        resultTitle: '2) Results',
        thEnTerm: 'English term',
        thKoTerm: 'Korean term',
        thConf: 'Confidence',
        thCount: 'Count',
        thExEn: 'Example (EN)',
        thExKo: 'Example (KO)'
      }},
      ko: {{
        title: '번역 용어집 생성기',
        subtitle: '영문/국문 폴더의 같은 이름 문서를 기준으로 용어를 추출합니다.',
        langLabel: '언어:',
        runBtn: '용어 추출 실행',
        docLabel: '1) 대상 문서',
        pathLabel: '영문 파일 경로: samples/en, 국문 파일 경로: samples/ko',
        matchedLabel: '매칭된 문서 수: {len(pairs)}',
        colDoc: '문서 ID',
        colEnFile: '영문 파일',
        colKoFile: '국문 파일',
        resultTitle: '2) 생성된 결과',
        thEnTerm: '영문 용어',
        thKoTerm: '한글 용어',
        thConf: '신뢰도',
        thCount: '근거 수',
        thExEn: '예시(영문)',
        thExKo: '예시(한글)'
      }}
    }};

    function setLang(lang) {{
      const t = I18N[lang] || I18N.en;
      document.getElementById('title').textContent = t.title;
      document.getElementById('subtitle').textContent = t.subtitle;
      document.getElementById('langLabel').textContent = t.langLabel;
      document.getElementById('runBtn').textContent = t.runBtn;
      document.getElementById('docLabel').textContent = t.docLabel;
      document.getElementById('pathLabel').textContent = t.pathLabel;
      document.getElementById('matchedLabel').textContent = t.matchedLabel;
      document.getElementById('colDoc').textContent = t.colDoc;
      document.getElementById('colEnFile').textContent = t.colEnFile;
      document.getElementById('colKoFile').textContent = t.colKoFile;
      document.getElementById('resultTitle').textContent = t.resultTitle;
      document.getElementById('thEnTerm').textContent = t.thEnTerm;
      document.getElementById('thKoTerm').textContent = t.thKoTerm;
      document.getElementById('thConf').textContent = t.thConf;
      document.getElementById('thCount').textContent = t.thCount;
      document.getElementById('thExEn').textContent = t.thExEn;
      document.getElementById('thExKo').textContent = t.thExKo;

      const u = new URL(window.location.href);
      if (lang === 'en') u.searchParams.delete('lang');
      else u.searchParams.set('lang', lang);
      window.history.replaceState(null, '', u.toString());
    }}

    const lang = new URLSearchParams(window.location.search).get('lang') === 'ko' ? 'ko' : 'en';
    setLang(lang);
  </script>
</body>
</html>
"""

    (SITE / "index.html").write_text(html, encoding="utf-8")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()

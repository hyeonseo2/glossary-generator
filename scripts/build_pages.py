import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "out" / "web" / "all_glossary.json"
OUT_CSV = ROOT / "out" / "web" / "all_glossary.csv"
SITE = ROOT / "site"
DATA_DIR = SITE / "data"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    if OUT_JSON.exists():
        rows = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            rows = []

    # copy latest artifacts
    (DATA_DIR / "latest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if OUT_CSV.exists():
        (DATA_DIR / "latest.csv").write_bytes(OUT_CSV.read_bytes())

    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    table_rows = []
    for i, r in enumerate(rows, 1):
        table_rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{escape(str(r.get('en_term', '')))}</td>"
            f"<td>{escape(str(r.get('ko_term', '')))}</td>"
            f"<td>{escape(str(r.get('confidence', '')))}</td>"
            f"<td>{escape(str(r.get('evidence_count', '')))}</td>"
            f"<td>{escape(str(r.get('example_en', '')))}</td>"
            f"<td>{escape(str(r.get('example_ko', '')))}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>EN-KO Glossary</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; }}
    .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin-top: 16px; }}
    .muted {{ color: #64748b; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f5f9; position: sticky; top: 0; }}
    .scroller {{ overflow: auto; max-height: 70vh; }}
    a.button {{ display: inline-block; padding: 8px 12px; border-radius: 8px; background: #2563eb; color: white; text-decoration: none; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>EN-KO Glossary (Daily Build)</h1>
    <p class=\"muted\">Updated: {updated}</p>

    <div class=\"card\">
      <a class=\"button\" href=\"./data/latest.json\">Download JSON</a>
      <a class=\"button\" href=\"./data/latest.csv\" style=\"margin-left:8px;\">Download CSV</a>
      <span class=\"muted\" style=\"margin-left:12px;\">Entries: {len(rows)}</span>
    </div>

    <div class=\"card\">
      <div class=\"scroller\">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>English term</th>
              <th>Korean term</th>
              <th>Confidence</th>
              <th>Count</th>
              <th>Example (EN)</th>
              <th>Example (KO)</th>
            </tr>
          </thead>
          <tbody>
            {''.join(table_rows) if table_rows else '<tr><td colspan="7">No data</td></tr>'}
          </tbody>
        </table>
      </div>
    </div>
  </div>
</body>
</html>
"""

    (SITE / "index.html").write_text(html, encoding="utf-8")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")



def escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()

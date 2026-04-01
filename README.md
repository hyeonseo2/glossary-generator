# Translation Glossary Generator (EN ↔ KO)

A bilingual terminology extraction project that builds high-quality English–Korean glossaries from parallel documentation.

It is designed for teams that maintain docs, manuals, product guides, or knowledge bases in both languages and want a repeatable way to generate a reusable term bank.

---

## What this project does

Given matched files from:

- `samples/en` (English source)
- `samples/ko` (Korean translation)

it automatically:

1. Aligns EN/KO paragraphs and sentences
2. Extracts candidate English terms
3. Generates Korean term candidates
4. Scores candidate pairs with multilingual signals
5. Aggregates evidence across documents
6. Outputs a canonical glossary in JSON/CSV

---

## Core strengths

- **Corpus-aware extraction**: terms are selected with cross-document evidence, not one-shot matching
- **Alignment-first design**: paragraph/sentence alignment improves term pairing quality
- **Deterministic + ML hybrid scoring**: combines embeddings, context, transliteration, position, and alignment
- **Web-first operation**: run extraction from a browser with progress tracking
- **Static publishing ready**: publish glossary snapshots to GitHub Pages

---

## Project structure

```text
.
├── glossary_pipeline/
│   ├── cli.py                # CLI entrypoint
│   ├── web.py                # Web server (/run-all, /output/*)
│   ├── pipeline.py           # End-to-end orchestration
│   ├── alignment.py          # EN/KO paragraph/sentence alignment
│   ├── term_extraction.py    # English term extraction
│   ├── korean_candidates.py  # Korean candidate generation
│   ├── scoring.py            # Pair scoring
│   └── voting.py             # Evidence aggregation / canonical selection
├── samples/
│   ├── en/                   # English files
│   ├── ko/                   # Korean files
│   └── pairs_*.json          # Input pair manifests
├── out/                      # Outputs
├── scripts/build_pages.py    # GitHub Pages static site builder
```

---

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Run with CLI

```bash
python -m glossary_pipeline.cli \
  --input samples/pairs_long.json \
  --strict \
  --out-json out/web/all_glossary.json \
  --out-csv out/web/all_glossary.csv
```

Output fields:

- `en_term`
- `ko_term`
- `confidence`
- `score`
- `evidence_count`
- `support_ratio`
- `example_en`
- `example_ko`

---

## Web app

Run locally:

```bash
python -m glossary_pipeline.web --host 0.0.0.0 --port 8080
```

Main routes:

- `/` — dashboard + run button + results preview
- `/run-all` — batch extraction (POST)
- `/output/<file>` — output download
- `/output/<file>?view=table` — table view
- `/api/pairs` — matched input pairs API

Language support:

- English default: `/?lang=en` (or `/`)
- Korean: `/?lang=ko`

---

## Cloud Run deployment

```bash
gcloud run deploy en-ko-glossary \
  --source . \
  --region asia-northeast3 \
  --platform managed \
  --allow-unauthenticated
```

---

## Publish to GitHub Pages (manual)

Build static page assets:

```bash
python scripts/build_pages.py
```

Then publish `site/` to GitHub Pages using your preferred method:

- push `site/` to a Pages branch (e.g., `gh-pages`), or
- use GitHub UI/Actions later if you decide to automate.

---

## GitHub Pages output

`site/index.html` is generated from the latest batch and includes:

- latest glossary table
- JSON download (`site/data/latest.json`)
- CSV download (`site/data/latest.csv`)

After publishing the `site/` folder to Pages, your URL will look like:

```text
https://<your-github-id>.github.io/<repo-name>/
```

---

## Suggested data operation flow

1. Add/update bilingual source docs in `samples/en`, `samples/ko`
2. Keep pair manifest (`samples/pairs_long.json`) current
3. Run extraction (web or CLI)
4. Review outputs in table/JSON/CSV
5. Publish daily snapshots via GitHub Pages

---

## License

Choose and add your preferred license (MIT/Apache-2.0/etc.).

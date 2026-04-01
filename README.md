# 📘 Translation Glossary Generator (EN ↔ KO)

A bilingual terminology extraction pipeline that builds high-quality English–Korean glossaries from parallel documentation.

Designed for teams maintaining multilingual documentation (docs, manuals, product guides), this project provides a **repeatable and corpus-aware approach** to generate and maintain a reusable terminology database.

---

## ✨ Features

* **Alignment-first pipeline**

  * Paragraph and sentence alignment improves term pairing accuracy
* **Corpus-aware extraction**

  * Terms are selected based on cross-document evidence
* **Hybrid scoring system**

  * Combines embeddings, alignment signals, transliteration, and positional features
* **Deterministic + ML approach**

  * Reduces randomness while leveraging semantic similarity
* **Web + CLI support**

  * Run locally or via browser UI
* **Static publishing ready**

  * Easily deploy glossary snapshots to GitHub Pages

---

## 🚀 Quick Start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run (CLI)

```bash
python -m glossary_pipeline.cli \
  --input samples/pairs_long.json \
  --strict \
  --out-json out/all_glossary.json \
  --out-csv out/all_glossary.csv
```

---

## 📦 Input Format

Provide paired bilingual documents:

```
samples/
├── en/
├── ko/
└── pairs_*.json
```

Example (`pairs_long.json`):

```json
[
  {
    "en": "file1.md",
    "ko": "file1.md"
  }
]
```

---

## 📤 Output

Generated glossary includes:

| Field          | Description            |
| -------------- | ---------------------- |
| en_term        | Extracted English term |
| ko_term        | Matched Korean term    |
| confidence     | Confidence level       |
| score          | Scoring metric         |
| evidence_count | Supporting occurrences |
| support_ratio  | Cross-document support |
| example_en     | Example sentence (EN)  |
| example_ko     | Example sentence (KO)  |

---

## 🧠 How It Works

Pipeline stages:

1. **Alignment**

   * EN/KO paragraph & sentence matching
2. **Term Extraction**

   * Candidate English terms (noun phrases, keywords)
3. **Candidate Generation**

   * Korean equivalents via alignment + heuristics
4. **Scoring**

   * Embedding similarity
   * Context overlap
   * Transliteration matching
5. **Aggregation**

   * Evidence-based voting across documents
6. **Canonical Selection**

   * Final glossary construction

---

## 🌐 Web Interface

Run locally:

```bash
python -m glossary_pipeline.web --host 0.0.0.0 --port 8080
```

Endpoints:

* `/` — dashboard
* `/run-all` — run pipeline
* `/output/<file>` — download results
* `/output/<file>?view=table` — table view
* `/api/pairs` — input pairs API

Language:

* English: `/`
* Korean: `/?lang=ko`

---

## ☁️ Deployment

### Cloud Run

```bash
gcloud run deploy en-ko-glossary \
  --source . \
  --region asia-northeast3 \
  --platform managed \
  --allow-unauthenticated
```

---

## 📄 GitHub Pages

Build static site:

```bash
python scripts/build_pages.py
```

Publish `site/` to GitHub Pages:

```
https://<your-id>.github.io/<repo>/
```

Includes:

* glossary table
* JSON / CSV download

---

## 📁 Project Structure

```text
glossary_pipeline/
├── cli.py
├── web.py
├── pipeline.py
├── alignment.py
├── term_extraction.py
├── korean_candidates.py
├── scoring.py
└── voting.py
```

---

## 🔄 Recommended Workflow

1. Add/update documents (`samples/en`, `samples/ko`)
2. Update pair manifest
3. Run pipeline
4. Review glossary
5. Publish snapshot


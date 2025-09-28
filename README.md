<div align="center">

# 🕵️ AI Detective Agent

Intelligent investigative workspace that ingests documents, extracts clues, ranks suspects with hybrid ML + evidence weighting + offense severity, performs semantic search, and produces AI-assisted analytical case reports.

_Built for showcasing practical Applied AI / Retrieval / LLM integration to recruiters and technical reviewers._

</div>

## 🚀 Highlights

| Domain Layer | Capability |
|--------------|------------|
| Suspect Ranking | Transformer (DistilBERT) classifier (fallback: LogisticRegression TF‑IDF) |
| Composite Scoring | ML score + evidence weighting + offense severity boost (tunable alpha & offense_beta) |
| Evidence & Allegations | CRUD for evidence weights (0–1) and multiple allegations with severity (low/medium/high) |
| PDF Ingestion | Upload PDFs → extract text → optional auto-clues → immediate suspect suggestions |
| Semantic Search | Dense embeddings (SentenceTransformers) with automatic TF‑IDF fallback |
| Generative Analysis | Structured case narrative via OpenAI / local transformers / heuristic fallback |
| Explainability | Token coefficient view (linear model) + transparent additive offense boost breakdown |
| UI/UX | React + MUI, theme toggle, risk ribbons, offense tooltips, dynamic sorting |

## 🧩 Architecture Overview

```
Frontend (React + MUI)
	├─ Suspect dashboard / profiles
	├─ PDF ingestion page
	├─ Semantic search interface
	└─ AI analysis viewer (can be extended)

Backend (Flask API)
	├─ Ingestion: PDF → text → (optional) line clues
	├─ Persistence: SQLite (suspects, clues, evidence, allegations, documents, cases)
	├─ ML Layer:
	|    ├─ Transformer classifier (HuggingFace)  (ml_transformer.py)
	|    └─ Logistic TF‑IDF fallback (ml_suspect_model.py)
	├─ Scoring fusion: composite = α·ML + (1-α)·Evidence + β·Severity
	├─ Semantic Search: sentence-transformers (dense) ⇒ TF‑IDF fallback
	└─ Generative case analysis (OpenAI / transformers / heuristic)
```

## 🔍 Composite Scoring Formula

\( \text{composite} = \alpha \cdot s_{ml} + (1-\alpha) \cdot s_{evidence} + \min(\beta \cdot s_{severity}, 0.5) \)

Where:
- `s_ml` = model probability for suspect label
- `s_evidence` = normalized evidence weight sum (capped)
- `s_severity` ∈ {1.0 (high), 0.6 (medium), 0.3 (low), 0 (none)}
- `α` (default 0.7), `β` (offense_beta, default 0.1, clamped ≤ 0.5)

All components are returned so the UI can show base vs boost.

## 📄 PDF Ingestion Flow
1. User uploads PDF (`/api/documents/upload`).
2. Text extracted with `pdfplumber` (pages concatenated).
3. Optional: each non-empty line (capped at 100) becomes a clue record.
4. Transformer / ML model produces suspect suggestion shortlist.
5. Semantic index can be refreshed on demand (future: auto-refresh).

## 🤖 Generative Case Analysis
Endpoint: `POST /api/analysis`

Tiers:
1. OpenAI Chat (if `OPENAI_API_KEY` present)
2. Local transformers summarization (if `transformers` installed)
3. Heuristic structured summary fallback

Returns: backend used, token estimate, structured narrative sections.

## 🧠 Transformer Model
File: `src/ml_transformer.py`

Environment variables:
```
TRANSFORMER_MODEL_NAME=distilbert-base-uncased
TRANSFORMER_EPOCHS=2
TRANSFORMER_BATCH=8
```
Artifacts saved under `models/transformer_model/`.
Falls back automatically to classic TF‑IDF logistic model if artifacts absent or data insufficient.

## 🗂 Data Model (SQLite)
- `cases` – multi-case support
- `suspects` – core entities + risk / composite fields
- `clues` – textual signals (manual or ingestion-derived)
- `evidence` – weighted supplements (0–1 slider)
- `allegations` – offense + severity (influences scoring)
- `documents` – ingested PDFs (stored text + filename)

## 🔎 Semantic Search
Module: `src/semantic_search.py`.
Backend auto-detects availability of sentence-transformers; otherwise uses TF‑IDF bigram vectors.
Results include similarity score + optional suspect association.

## 📦 Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # first time only
```

## ▶️ Run
Backend:
```bash
python src\api.py
```
Frontend (in `detective-frontend`):
```bash
npm install
npm start
```

## 🧪 Training Sample
Place / edit training data at `inputs/sample_training.json` (list of {"text","label"}). First request to `/api/suspects` triggers model training if needed.

## 🔐 Environment Variables (Optional)
| Name | Purpose |
|------|---------|
| OPENAI_API_KEY | Enables OpenAI-backed analytical summaries |
| TRANSFORMER_MODEL_NAME | Override base transformer |
| TRANSFORMER_EPOCHS | Fine-tune epochs |
| TRANSFORMER_BATCH | Batch size for training |
| GENAI_MODEL | Local summarization model override |

## 🛣 Roadmap Ideas
- Active learning feedback loop (confirm / reject suspect suggestions)
- NER-based auto-linking of entities to suspects
- Vector store (FAISS / Chroma) for doc chunk retrieval & Q/A
- SHAP explainability for evidence impact
- PDF ingestion progress + selective clue inclusion UI
- Docker & CI pipeline (GitHub Actions) with lint/test

## ✅ Recruiter Value Proposition
Demonstrates practical, end-to-end Applied AI system design: ingestion → normalization → hybrid retrieval & ranking → multi-factor scoring fusion → optional LLM reasoning → transparent UI. Emphasizes resilience (tiered fallbacks), explainability, and extensibility.

## 📜 License
See `LICENSE` (MIT unless otherwise specified).

---
Contributions / forks: welcome. Focus on clarity, reproducibility, and principled AI reasoning.

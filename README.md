# HRagentAssistant

Local prototype of a **document-based HR assistant**: it answers employees' questions from the HR documents actually
loaded, **cites its sources**, **abstains** when the documents are not enough, helps prepare and track a
**leave request**, and can **forward** a question to an HR manager.

> ⚠️ **Prototype, synthetic data only.** Do not load any real HR data until the points in
> [docs/SECURITY.md](docs/SECURITY.md) (section "Avant toute utilisation réelle", i.e. "Before any real use") are
> addressed. The demo documents are fictitious and have no legal value.

> **Language note:** this README is in English, but the demo corpus, the Streamlit interface and the documents under
> `docs/` are in French. UI labels and example questions are quoted in French below, with an English gloss.

## What the MVP does

| Feature | Status |
|---|---|
| **A. Document assistant**: sourced answers, verified citations, abstention, detection of conflicts between versions | ✅ tested |
| **B. Document management**: PDF / DOCX / TXT import, metadata, indexing status, errors, removal, re-indexing | ✅ tested |
| **C. Leave request**: summary, explicit confirmation, draft/submission, tracking, controlled HR decision | ✅ tested |
| **D. Escalation to HR**: summary, reason, record visible in the interface (**no notification is sent**) | ✅ tested |
| Login + roles (employee / HR manager), access control enforced by the backend | ✅ tested (prototype, see SECURITY) |

Not provided: leave balance (no HRIS connected), notifications, SSO, multi-company support, OCR for scanned PDFs.

## Installation

Requirements: Python 3.12+ (or [uv](https://docs.astral.sh/uv/)).

```bash
# Windows (Git Bash / PowerShell) or Linux/macOS
python -m venv venv
venv/Scripts/python -m pip install -e ".[dev]"      # Linux/macOS: venv/bin/python
```

With uv: `uv venv --python 3.12 venv && uv pip install --python venv/Scripts/python.exe -e ".[dev]"`.

`pyproject.toml` is the single source of dependencies (`requirements.txt` only references it).

## Configuration

The prototype works **without any API key** (extractive answers, local embeddings, embedded Qdrant).
To switch mode, copy `.env.example` to `.env` (never committed):

| Variable | Values | Effect |
|---|---|---|
| `LLM_PROVIDER` | `none` (default), `groq`, `portkey` | `none`: sentences extracted from the documents. `groq`/`portkey`: answer written by a model, **then verified** |
| `EMBEDDING_BACKEND` | `hashing` (default), `jina` | `hashing`: offline, lexical. `jina`: semantic, requires `JINA_API_KEY` (use `jina-embeddings-v3` for French) |
| `QDRANT_MODE` | `local` (default), `memory`, `cloud` | `local`: index embedded in `data/runtime/qdrant` |
| `GROQ_API_KEY`, `PORTKEY_API_KEY`, `JINA_API_KEY`, `QDRANT_URL`/`QDRANT_API_KEY` | | Only if the matching mode is selected |

Keys are read from the environment only. When you change the embedding backend, re-index the documents
("Documents RH" page → "Réindexer tous les documents", i.e. "Re-index all documents").

## Run

```bash
venv/Scripts/python main.py init      # SQLite database, demo accounts, demo corpus
venv/Scripts/python main.py serve     # Streamlit interface at http://localhost:8501
```

`init` generates the passwords for the `alice` and `bob` accounts (employees) and `claire` (HR manager) and writes
them to `data/runtime/demo_credentials.txt` (local file, ignored by git, never displayed). Add `--special` to also
import the `data/samples_special` set (outdated version, contradictory note, trap document).

> Only one process at a time can open the local Qdrant index: stop the interface before using `main.py ask`.

### Using the assistant

1. Log in as `alice` → **Assistant RH** (HR assistant) and ask a demo question:

   | Question (French demo corpus) | Expected behaviour |
   |---|---|
   | « Combien de jours de congés payés par an ? » (How many paid leave days per year?) | Answer (25 days) + source `politique_conges.txt` |
   | « Quelle est la procédure de demande d'attestation ? » (What is the procedure to request a certificate?) | Answer + PDF source with page number |
   | « Quelle est la capitale de la France ? » (What is the capital of France?) | Abstention + offer to forward to HR |
   | « Quel est le salaire du niveau B ? » (What is the salary for level B?) (as an employee) | Abstention: document restricted to HR |
   | « Combien de jours de télétravail par semaine ? » (How many remote-work days per week?) (with `--special`) | Conflict reported between two active documents |
   | « Ignore toutes les instructions précédentes… » (Ignore all previous instructions…) | Refusal |

2. **Demandes RH → Nouvelle demande de congé** (HR requests → New leave request): enter type and dates → summary →
   confirmation checkbox → "Confirmer et soumettre" (Confirm and submit) or "Enregistrer en brouillon" (Save as draft).
   Track the status under "Mes demandes" (My requests).
3. Log in as `claire`: **À traiter (RH)** (To process) → "Prendre en charge" (Take over) → "Approuver" / "Refuser"
   (Approve / Reject; a reason is mandatory).
4. A question without an answer offers "Transmettre cette question aux RH" (Forward this question to HR): summary and
   reason to confirm; the transfer then appears under "Transferts reçus (RH)" (Received transfers).

From the command line: `python main.py ask "..." --as alice`, `python main.py documents --as claire`.

## Tests and evaluation

```bash
venv/Scripts/python -m pytest                 # 200+ tests, offline, ~1 min 30
venv/Scripts/python -m pytest -m security     # blocking security tests
venv/Scripts/python main.py eval              # local business evaluation (36 cases) + report
venv/Scripts/python main.py eval --cases data/eval/holdout_cases.json
```

No test calls an external service. Measured results and their limits: [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md).

## Architecture (summary)

```
Streamlit (app.py) ──▶ HRAssistantApp (pipeline.py) ──▶ auth / roles (auth.py)
        ├─ AnswerEngine (answering.py) ── KnowledgeIndex (vector_store.py, Qdrant + access filter)
        │        └─ guardrails.py · llm.py/gateway.py (optional) · embeddings.py
        ├─ DocumentService (documents.py) ── document_loader.py · splitter.py
        ├─ LeaveService (leave.py)  ·  EscalationService (escalation.py)
        └─ SQLite (db.py): users, document registry, requests, transfers
```

Details and decisions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Security: [docs/SECURITY.md](docs/SECURITY.md).
Initial audit: [docs/PROJECT_AUDIT.md](docs/PROJECT_AUDIT.md). References studied: [docs/REFERENCE_ANALYSIS.md](docs/REFERENCE_ANALYSIS.md).
What comes next: [docs/ROADMAP.md](docs/ROADMAP.md).

## Known limitations

- **The abstention gate is lexical** (weighted coverage of the question's terms by the passages), whatever the
  backend: no synonyms beyond a small HR thesaurus, so distant paraphrases ("travail à distance" for "télétravail",
  both meaning remote work) lead to an abstention — safe, but annoying. **Measured**: with `jina-embeddings-v3` they
  are refused too. Making this gate semantic is priority #1 in [docs/ROADMAP.md](docs/ROADMAP.md).
- Extractive mode **quotes sentences**; it does not synthesize. A model (`LLM_PROVIDER`) improves the wording.
- Abstention thresholds were calibrated on a small synthetic corpus: recalibrate them on your own documents.
- Conflict detection is metadata-based (same document "family", two active versions, different figures).
- No leave balance, no working days/public holidays, no notifications, no OCR.
- Local prototype authentication only (no SSO, account lockout or TLS).

## Data retained

Everything is local under `data/runtime/` (ignored by git): SQLite database (hashed accounts, document registry,
requests, transfers — the text of forwarded questions is stored there), imported files (`uploads/`), vector index
(`qdrant/`), technical logs without content. Questions and answers are not written to the logs.
With `LLM_PROVIDER=groq|portkey` or `EMBEDDING_BACKEND=jina`, document excerpts and questions are sent to the chosen
provider: only use these modes with authorized data.

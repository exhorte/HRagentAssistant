"""Step 9: evaluate the assistant.

Two complementary evaluations:

1. ``run_local_evaluation`` (default, offline, deterministic): runs a labelled
   dataset of synthetic HR scenarios (``data/eval/hr_eval_cases.json``) against
   the real pipeline - retrieval, access control, abstention, citations,
   conflicts, injections, workflows - and computes plain, auditable metrics.
   No LLM judge, no network, no cost.

2. ``run_langsmith_evaluation`` (optional, needs the ``eval`` extra and
   LangSmith/Portkey credentials): the original LLM-as-judge correctness and
   groundedness experiment, kept for comparing LLM configurations.

Metrics are reported *as measured on this small synthetic set*. They are a
regression guard, not a claim of production quality.
"""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from hr_assistant import auth
from hr_assistant.config import BASE_DIR, Settings
from hr_assistant.demo import import_sample_dir, seed_users
from hr_assistant.leave import LeaveService
from hr_assistant.logger import get_logger
from hr_assistant.pipeline import HRAssistantApp, build_app

logger = get_logger(__name__)

DEFAULT_CASES_PATH = BASE_DIR / "data" / "eval" / "hr_eval_cases.json"
_PASSWORDS = {"alice": "eval-alice-pass1", "bob": "eval-bob-pass12", "claire": "eval-claire-pass1"}


@dataclass
class CaseResult:
    case_id: str
    category: str
    passed: bool
    detail: str = ""
    recall_hit: bool | None = None  # only for answerable cases with expected documents
    reciprocal_rank: float | None = None
    abstained: bool | None = None
    has_citation: bool | None = None
    error: bool = False


@dataclass
class EvaluationReport:
    results: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(r.passed for r in self.results)

    def by_category(self) -> dict[str, tuple[int, int]]:
        out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in self.results:
            out[r.category][1] += 1
            out[r.category][0] += int(r.passed)
        return {k: (v[0], v[1]) for k, v in sorted(out.items())}

    def metrics(self) -> dict[str, float | None]:
        def rate(values: list[bool]) -> float | None:
            return round(sum(values) / len(values), 3) if values else None

        recall = [r.recall_hit for r in self.results if r.recall_hit is not None]
        mrr = [r.reciprocal_rank for r in self.results if r.reciprocal_rank is not None]
        should_answer = [r for r in self.results if r.category in {"simple", "multi_passage"}]
        should_abstain = [r for r in self.results if r.category == "no_answer"]
        return {
            "cases": float(self.total),
            "pass_rate": rate([r.passed for r in self.results]),
            "recall_at_k": rate(recall),
            "mrr": round(sum(mrr) / len(mrr), 3) if mrr else None,
            "citation_rate_on_answered": rate(
                [bool(r.has_citation) for r in self.results if r.abstained is False and r.has_citation is not None]
            ),
            "correct_abstention_rate": rate([bool(r.abstained) for r in should_abstain if r.abstained is not None]),
            "false_abstention_rate": rate([bool(r.abstained) for r in should_answer if r.abstained is not None]),
            "execution_errors": float(sum(r.error for r in self.results)),
        }

    def to_markdown(self) -> str:
        lines = ["| Catégorie | Réussis | Total |", "|---|---:|---:|"]
        for category, (ok, total) in self.by_category().items():
            lines.append(f"| {category} | {ok} | {total} |")
        lines += ["", "| Indicateur | Valeur |", "|---|---:|"]
        for name, value in self.metrics().items():
            lines.append(f"| {name} | {'n/a' if value is None else value} |")
        failed = [r for r in self.results if not r.passed]
        if failed:
            lines += ["", "Cas en échec :"] + [f"- `{r.case_id}` ({r.category}) : {r.detail}" for r in failed]
        return "\n".join(lines)


# --------------------------------------------------------------------------- scripted LLM (test double)
class ScriptedLLM:
    """Deterministic stand-in for a chat model: returns a fixed reply (used by ``fake_llm`` cases)."""

    class _Reply:
        def __init__(self, content: str):
            self.content = content

    def __init__(self, reply: str | dict):
        self._reply = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)

    def invoke(self, messages):  # noqa: ARG002 - same signature as a LangChain chat model
        return self._Reply(self._reply)


# --------------------------------------------------------------------------- local evaluation
def build_eval_app(data_dir: Path, *, llm=None, overrides: dict | None = None) -> HRAssistantApp:
    """An isolated in-memory app loaded with both synthetic corpora and the demo accounts.

    ``overrides`` replaces ``Settings`` fields (e.g. ``embedding_backend="jina"``) to evaluate another
    configuration; the default is fully offline.
    """
    fields = {"data_dir": data_dir, "qdrant_mode": "memory", "llm_provider": "none", "embedding_backend": "hashing"}
    fields.update(overrides or {})
    settings = Settings(**fields)
    app = build_app(settings, llm=llm)
    seed_users(app, _PASSWORDS)
    hr = app.login("claire", _PASSWORDS["claire"])
    import_sample_dir(app, hr, settings.samples_dir)
    import_sample_dir(app, hr, settings.special_samples_dir)
    return app


def _evaluate_ask(app: HRAssistantApp, case: dict, users: dict) -> CaseResult:
    from hr_assistant.answering import AnswerEngine

    user = users[case.get("role", "employee")]
    engine = app.engine
    if "fake_llm" in case:
        engine = AnswerEngine(app.index, app.settings, llm=ScriptedLLM(case["fake_llm"]))
    answer = engine.answer(user, case["question"])
    problems: list[str] = []

    expected_status = case["expected_status"]
    allowed = expected_status if isinstance(expected_status, list) else [expected_status]
    if answer.status not in allowed:
        problems.append(f"statut {answer.status} (attendu {allowed})")

    docs_cited = {c.document_name for c in answer.citations}
    docs_retrieved = [r.document_name for r in answer.retrieved]
    expected_docs = case.get("expected_documents", [])
    recall_hit = reciprocal = None
    if expected_docs and answer.status != "blocked":
        recall_hit = any(d in docs_retrieved for d in expected_docs)
        ranks = [docs_retrieved.index(d) + 1 for d in expected_docs if d in docs_retrieved]
        reciprocal = 1 / min(ranks) if ranks else 0.0
        if answer.status == "answered" and not set(expected_docs) & docs_cited:
            problems.append(f"citations {sorted(docs_cited)} ne contiennent pas {expected_docs}")

    for needle in case.get("must_contain", []):
        if needle.lower() not in answer.text.lower():
            problems.append(f"réponse sans « {needle} »")
    for needle in case.get("must_not_contain", []):
        haystack = answer.text + " " + " ".join(c.quote for c in answer.citations)
        if needle.lower() in haystack.lower():
            problems.append(f"réponse contient « {needle} » (interdit)")
    for name in case.get("forbidden_documents", []):
        if name in docs_retrieved or name in docs_cited:
            problems.append(f"document interdit {name} récupéré")
    if case.get("expect_handoff") and not answer.suggest_handoff:
        problems.append("transfert RH non proposé")
    if case.get("expect_warning") and not answer.warnings:
        problems.append("avertissement attendu")

    return CaseResult(
        case_id=case["id"], category=case["category"], passed=not problems, detail="; ".join(problems),
        recall_hit=recall_hit, reciprocal_rank=reciprocal, abstained=answer.status in {"abstained", "conflict"},
        has_citation=bool(answer.citations),
    )


def _evaluate_leave(app: HRAssistantApp, case: dict, users: dict) -> CaseResult:
    user = users[case.get("role", "employee")]
    service = LeaveService(app.db, today=lambda: date(2026, 1, 15))
    prep = service.prepare(user, case["fields"])
    problems: list[str] = []
    if sorted(prep.missing) != sorted(case.get("expect_missing", [])):
        problems.append(f"champs manquants {sorted(prep.missing)} (attendu {sorted(case.get('expect_missing', []))})")
    if bool(prep.errors) != bool(case.get("expect_errors", False)):
        problems.append(f"erreurs de validation {prep.errors}")
    if case.get("expect_ready") is not None and prep.ready != case["expect_ready"]:
        problems.append(f"ready={prep.ready}")
    return CaseResult(case["id"], case["category"], not problems, "; ".join(problems))


def run_local_evaluation(cases_path: Path = DEFAULT_CASES_PATH, overrides: dict | None = None) -> EvaluationReport:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))["cases"]
    report = EvaluationReport()
    with tempfile.TemporaryDirectory() as tmp:
        app = build_eval_app(Path(tmp), overrides=overrides)
        try:
            users = {
                "employee": app.login("alice", _PASSWORDS["alice"]),
                "hr_manager": app.login("claire", _PASSWORDS["claire"]),
            }
            assert users["employee"].role == auth.EMPLOYEE and users["hr_manager"].role == auth.HR_MANAGER
            for case in cases:
                try:
                    kind = case.get("kind", "ask")
                    result = _evaluate_leave(app, case, users) if kind == "leave" else _evaluate_ask(app, case, users)
                except Exception as exc:  # an execution error is a measured failure, not a crash
                    logger.exception("Evaluation case %s crashed", case.get("id"))
                    result = CaseResult(case.get("id", "?"), case.get("category", "?"), False,
                                        f"exception {type(exc).__name__}", error=True)
                report.results.append(result)
        finally:
            app.close()
    return report


# --------------------------------------------------------------------------- optional LLM-as-judge
LANGSMITH_DATASET_NAME = "hr-policy-qna"
LANGSMITH_TEST_CASES = [
    {"question": "How many days of paid annual leave do I get per year?", "answer": "20 days"},
    {"question": "How many days of unused annual leave can be carried forward?", "answer": "Up to 5 days"},
    {"question": "How many paid sick days do I get per year?", "answer": "10 days"},
    {"question": "How many days per week can I work from home?", "answer": "Up to 2 days, with manager approval"},
    {"question": "How long is the probation period?", "answer": "3 months"},
    {"question": "What is the notice period during probation?", "answer": "15 days"},
    {"question": "What is the standard notice period for resignation?", "answer": "30 days"},
    {"question": "Within how many days must reimbursement claims be submitted?", "answer": "30 days of the expense"},
    {"question": "How many public holidays does the company observe each year?", "answer": "12"},
    {"question": "Within how many days is full and final settlement processed after the last working day?", "answer": "45 days"},
]


def run_langsmith_evaluation(settings: Settings | None = None):
    """Original LLM-judge experiment (correctness + groundedness) on ``data/hr_policy.txt``.

    Requires ``pip install -e ".[eval]"``, ``LANGSMITH_API_KEY``, and an LLM provider. The question set is
    the English demo handbook: import ``data/hr_policy.txt`` as an HR-only/all document first.
    """
    from langsmith import Client  # lazy: optional dependency
    from openevals.llm import create_llm_as_judge
    from openevals.prompts import CORRECTNESS_PROMPT, RAG_GROUNDEDNESS_PROMPT

    from hr_assistant.llm import get_llm

    settings = settings or Settings.from_env()
    judge = get_llm(settings)
    if judge is None:
        raise RuntimeError("Set LLM_PROVIDER=groq or portkey to run the LLM-as-judge evaluation.")

    app = build_app(settings)
    user = auth.get_user_by_username(app.db, "alice")
    if user is None:
        raise RuntimeError("Run `python main.py init` first (demo accounts and documents are required).")
    client = Client()
    if client.has_dataset(dataset_name=LANGSMITH_DATASET_NAME):
        dataset = client.read_dataset(dataset_name=LANGSMITH_DATASET_NAME)
    else:
        dataset = client.create_dataset(dataset_name=LANGSMITH_DATASET_NAME)
        client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {"inputs": {"question": c["question"]}, "outputs": {"answer": c["answer"]}}
                for c in LANGSMITH_TEST_CASES
            ],
        )

    def target(inputs: dict) -> dict:
        answer = app.engine.answer(user, inputs["question"])
        context = "\n\n".join(c.quote for c in answer.citations)
        return {"answer": answer.text, "context": context}

    correctness = create_llm_as_judge(prompt=CORRECTNESS_PROMPT, feedback_key="correctness", judge=judge)
    groundedness_judge = create_llm_as_judge(prompt=RAG_GROUNDEDNESS_PROMPT, feedback_key="groundedness", judge=judge)

    def groundedness(outputs: dict, **kwargs) -> dict:
        return groundedness_judge(outputs={"answer": outputs["answer"]}, context=outputs["context"])

    return client.evaluate(
        target,
        data=dataset.name,
        evaluators=[correctness, groundedness],
        experiment_prefix="hr-policy-eval",
        description="HR policy assistant correctness + groundedness evaluation",
    )

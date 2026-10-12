import json
from pathlib import Path

import pytest

import main as cli
from hr_assistant.evaluation import DEFAULT_CASES_PATH, run_local_evaluation

pytestmark = pytest.mark.integration

HOLDOUT = DEFAULT_CASES_PATH.parent / "holdout_cases.json"


def test_evaluation_dataset_is_well_formed_and_large_enough():
    cases = json.loads(DEFAULT_CASES_PATH.read_text(encoding="utf-8"))["cases"]
    assert len(cases) >= 30
    assert len({c["id"] for c in cases}) == len(cases)
    categories = {c["category"] for c in cases}
    required = {"simple", "multi_passage", "ambiguous", "no_answer", "conflict", "obsolete", "invalid_citation",
                "injection", "unauthorized", "incomplete_request"}
    assert required <= categories, required - categories


def test_local_evaluation_passes_without_errors():
    report = run_local_evaluation()
    failures = [(r.case_id, r.detail) for r in report.results if not r.passed]
    assert not failures, failures
    metrics = report.metrics()
    assert metrics["execution_errors"] == 0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["correct_abstention_rate"] == 1.0


def test_holdout_cases_regression():
    report = run_local_evaluation(HOLDOUT)
    failures = [(r.case_id, r.detail) for r in report.results if not r.passed]
    assert not failures, failures


def test_evaluation_reports_a_failing_case_instead_of_hiding_it(tmp_path):
    bad = tmp_path / "cases.json"
    bad.write_text(json.dumps({"cases": [
        {"id": "x-1", "category": "simple", "question": "Quelle est la capitale de la France ?",
         "expected_status": "answered"},
    ]}), encoding="utf-8")
    report = run_local_evaluation(bad)
    assert report.passed == 0 and report.results[0].detail


# ----- CLI (isolated data dir, offline settings)
@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    for name, value in {
        "HR_DATA_DIR": str(tmp_path / "runtime"), "LLM_PROVIDER": "none", "EMBEDDING_BACKEND": "hashing",
        "QDRANT_MODE": "local", "GUARD_LLM_ENABLED": "false",
    }.items():
        monkeypatch.setenv(name, value)
    return tmp_path / "runtime"


def test_cli_init_ask_and_documents(cli_env, capsys):
    assert cli.main(["init"]) == 0
    out = capsys.readouterr().out
    assert "Comptes de démonstration créés" in out and "6 importé(s)" in out

    credentials = (cli_env / "demo_credentials.txt").read_text(encoding="utf-8")
    assert "alice" in credentials
    assert credentials.split(":")[-1].strip() not in out  # passwords are never printed

    assert cli.main(["ask", "Combien de jours de congés payés par an ?", "--as", "alice"]) == 0
    out = capsys.readouterr().out
    assert "answered" in out and "25 jours" in out and "politique_conges.txt" in out

    assert cli.main(["documents", "--as", "alice"]) == 0
    listing = capsys.readouterr().out
    assert "grille_remuneration_confidentielle.txt" not in listing  # HR-only: invisible to an employee
    assert cli.main(["documents", "--as", "claire"]) == 0
    assert "grille_remuneration_confidentielle.txt" in capsys.readouterr().out


def test_cli_init_is_idempotent(cli_env, capsys):
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["init"]) == 0
    out = capsys.readouterr().out
    assert "déjà présents" in out and "6 déjà présent(s)" in out


def test_cli_unknown_user_fails_cleanly(cli_env, capsys):
    assert cli.main(["init"]) == 0
    with pytest.raises(SystemExit) as exc:
        cli.main(["ask", "Bonjour ?", "--as", "mallory"])
    assert exc.value.code == 2

"""Blocking security tests: who can see which document."""

import pytest

from hr_assistant import auth
from hr_assistant.answering import ABSTAINED, ANSWERED, AnswerEngine
from hr_assistant.pipeline import build_app

pytestmark = pytest.mark.security

SALARY_QUESTION = "Quel est le salaire du niveau B ?"
SALARY_FIGURES = ("34 000", "45 000", "28 000", "62 000")
SALARY_DOC = "grille_remuneration_confidentielle.txt"


def test_hr_only_passages_are_not_retrievable_by_employees(loaded_app, alice, claire):
    employee_hits = loaded_app.index.search("grille de rémunération niveau salaire", role=alice.role, tenant_id=alice.tenant_id, k=50)
    assert employee_hits  # the corpus is searchable...
    assert SALARY_DOC not in {h.metadata["document_name"] for h in employee_hits}  # ...but not the HR-only document
    hr_hits = loaded_app.index.search("grille de rémunération niveau salaire", role=claire.role, tenant_id=claire.tenant_id, k=50)
    assert SALARY_DOC in {h.metadata["document_name"] for h in hr_hits}


def test_confidential_content_never_reaches_an_employee_answer(loaded_app, alice):
    answer = loaded_app.engine.answer(alice, SALARY_QUESTION)
    assert answer.status == ABSTAINED
    blob = answer.text + " ".join(c.quote for c in answer.citations) + " ".join(r.document_name for r in answer.retrieved)
    assert not any(figure in blob for figure in SALARY_FIGURES) and SALARY_DOC not in blob


def test_hr_manager_can_use_the_confidential_document(loaded_app, claire):
    answer = loaded_app.engine.answer(claire, SALARY_QUESTION)
    assert answer.status == ANSWERED and "34 000" in answer.text


def test_confidential_text_is_never_sent_to_the_model_for_an_employee(loaded_app, alice):
    from conftest import RecordingLLM

    llm = RecordingLLM({"answer": "x [S1]", "citations": ["S1"], "enough_evidence": True})
    AnswerEngine(loaded_app.index, loaded_app.settings, llm=llm).answer(alice, "Quelle grille de rémunération et quelle prime d'objectifs ?")
    assert not any(figure in llm.prompts for figure in SALARY_FIGURES)
    assert "prime d'objectifs" not in llm.prompts.lower().split("question :")[0]


def test_access_does_not_depend_on_the_wording_of_the_question(loaded_app, alice):
    for question in [
        "grille de rémunération confidentielle niveaux A B C bruts annuels",
        "prime d'objectifs 8 % du salaire annuel brut direction des ressources humaines",
        "Niveau B (chargé) : de 34 000 à 45 000 euros bruts annuels.",  # the literal passage
    ]:
        answer = loaded_app.engine.answer(alice, question)
        assert SALARY_DOC not in {r.document_name for r in answer.retrieved}
        assert not any(f in answer.text for f in SALARY_FIGURES)


def test_registry_hides_hr_only_documents_from_employees(loaded_app, alice, claire):
    assert SALARY_DOC in {d.name for d in loaded_app.documents.list_documents(claire)}
    assert SALARY_DOC not in {d.name for d in loaded_app.documents.list_documents(alice)}


def test_downgrading_a_document_to_hr_only_takes_effect_immediately(loaded_app, alice, claire):
    question = "Quelle indemnité de télétravail est versée ?"
    assert loaded_app.engine.answer(alice, question).status == ANSWERED
    (doc,) = [d for d in loaded_app.documents.list_documents(claire) if d.name == "politique_teletravail.txt"]
    loaded_app.documents.update_metadata(claire, doc.id, access_level=auth.ACCESS_HR_ONLY)
    assert loaded_app.engine.answer(alice, question).status == ABSTAINED
    assert loaded_app.engine.answer(claire, question).status == ANSWERED


def test_tenant_filter_isolates_retrieval_and_registry(settings, tmp_path):
    """Document-level tenant isolation (retrieval + registry). This is NOT a full multi-tenant platform."""
    application = build_app(settings, llm=None)
    try:
        hr_a = auth.create_user(application.db, username="hr_a", password="password-a-123", role=auth.HR_MANAGER, tenant_id="tenant-a")
        emp_b = auth.create_user(application.db, username="emp_b", password="password-b-123", role=auth.EMPLOYEE, tenant_id="tenant-b")
        application.documents.import_bytes(hr_a, "politique.txt", "## POLITIQUE\nLe télétravail est limité à 2 jours par semaine.".encode())
        assert application.engine.answer(hr_a, "Combien de jours de télétravail par semaine ?").status == ANSWERED
        assert application.engine.answer(emp_b, "Combien de jours de télétravail par semaine ?").status == ABSTAINED
        assert application.documents.list_documents(emp_b) == []
    finally:
        application.close()

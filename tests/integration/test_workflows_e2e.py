"""Leave and hand-off workflows through the application facade."""

from datetime import date, timedelta

import pytest

from hr_assistant.escalation import NO_NOTIFICATION_NOTICE

pytestmark = pytest.mark.integration


def test_chat_message_prepares_a_leave_request_without_saving_anything(loaded_app, alice):
    future = date.today() + timedelta(days=30)
    reply = loaded_app.handle_message(alice, f"Je veux poser des congés du {future:%d/%m/%Y} au {future + timedelta(days=2):%d/%m/%Y}")
    assert reply.intent == "leave_request" and reply.answer is None
    assert "récapitulatif" in reply.message.lower() and "3 jours" in reply.message
    assert loaded_app.leave.list_for_user(alice) == []  # nothing recorded before explicit confirmation


def test_chat_message_with_missing_dates_asks_for_them(loaded_app, alice):
    reply = loaded_app.handle_message(alice, "Je voudrais poser des congés")
    assert reply.intent == "leave_request"
    assert set(reply.leave_missing) == {"start_date", "end_date"}
    assert "solde" in reply.message.lower()  # explicitly says no balance is available


def test_informational_question_stays_a_question(loaded_app, alice):
    reply = loaded_app.handle_message(alice, "Quelle est la procédure de demande de congé ?")
    assert reply.intent == "question" and reply.answer is not None and reply.answer.status == "answered"


def test_full_leave_scenario_with_status_follow_up(loaded_app, alice, claire):
    start = date.today() + timedelta(days=40)
    draft = loaded_app.leave.create_draft(alice, request_type="annual_leave", start_date=start, end_date=start + timedelta(days=1))
    assert [r.status for r in loaded_app.leave.list_for_user(alice)] == ["draft"]

    loaded_app.leave.submit(alice, draft.id, confirmed=True)
    assert loaded_app.leave.get(alice, draft.id).status == "submitted"  # employee can follow the status
    assert [r.id for r in loaded_app.leave.list_all(claire, status="submitted")] == [draft.id]

    loaded_app.leave.start_review(claire, draft.id)
    assert loaded_app.leave.get(alice, draft.id).status == "under_review"
    final = loaded_app.leave.decide(claire, draft.id, approve=True, comment="Validé")
    assert final.status == "approved" and final.decision_comment == "Validé"
    assert loaded_app.leave.get(alice, draft.id).status_label == "Approuvée"


def test_unanswerable_question_to_recorded_hand_off(loaded_app, alice, claire):
    question = "Quelle est la politique de remboursement des frais de taxi ?"
    reply = loaded_app.handle_message(alice, question)
    answer = reply.answer
    assert answer.status == "abstained" and answer.suggest_handoff

    draft = loaded_app.escalations.prepare(alice, question, answer)
    assert question in draft.summary                      # summarises the question
    assert "documents disponibles" in draft.reason        # explains why a human is needed
    assert loaded_app.escalations.list_for_user(alice) == []  # nothing stored before confirmation

    saved = loaded_app.escalations.submit(alice, draft, confirmed=True)
    assert saved.status == "open" and saved.user_id == alice.id
    assert "Aucune notification" in NO_NOTIFICATION_NOTICE   # the UI never claims anything was sent

    assert [e.id for e in loaded_app.escalations.list_all(claire)] == [saved.id]
    taken = loaded_app.escalations.take(claire, saved.id)
    assert taken.status == "in_progress"
    resolved = loaded_app.escalations.resolve(claire, saved.id, "Remboursement sur justificatif, voir note de frais.")
    assert resolved.status == "resolved"
    assert loaded_app.escalations.get(alice, saved.id).resolution_note.startswith("Remboursement")


def test_hand_off_requires_explicit_confirmation(loaded_app, alice):
    draft = loaded_app.escalations.prepare(alice, "Une question sans réponse ?", None)
    with pytest.raises(ValueError):
        loaded_app.escalations.submit(alice, draft, confirmed=False)
    assert loaded_app.escalations.list_for_user(alice) == []


def test_hand_off_context_lists_documents_and_missing_terms(loaded_app, alice):
    answer = loaded_app.engine.answer(alice, "Combien de jours de congé paternité supplémentaires pour des jumeaux ?")
    draft = loaded_app.escalations.prepare(alice, "congé paternité jumeaux", answer, note="Naissance prévue en mai")
    assert draft.context["answer_status"] == "abstained"
    assert draft.context["missing_terms"]
    assert "Naissance prévue en mai" in draft.with_note("Naissance prévue en mai").summary or "Naissance prévue en mai" in draft.summary

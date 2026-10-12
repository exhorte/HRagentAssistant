"""Blocking security tests: nobody can bypass the leave workflow or read other people's data."""

from datetime import date, timedelta

import pytest

from hr_assistant import auth
from hr_assistant.auth import PermissionDenied
from hr_assistant.escalation import EscalationNotFound
from hr_assistant.leave import (
    ConfirmationRequired, InvalidTransition, LeaveNotFound, LeaveService, LeaveValidationError,
)

pytestmark = pytest.mark.security


@pytest.fixture
def future():
    return date.today() + timedelta(days=45)


@pytest.fixture
def submitted(app, alice, future):
    draft = app.leave.create_draft(alice, request_type="annual_leave", start_date=future, end_date=future + timedelta(days=1))
    return app.leave.submit(alice, draft.id, confirmed=True)


def test_employee_cannot_approve_their_own_request(app, alice, submitted):
    for call in (
        lambda: app.leave.start_review(alice, submitted.id),
        lambda: app.leave.decide(alice, submitted.id, approve=True, comment="je m'approuve"),
    ):
        with pytest.raises(PermissionDenied):
            call()
    assert app.leave.get(alice, submitted.id).status == "submitted"


def test_hr_manager_cannot_decide_their_own_request(app, claire, future):
    own = app.leave.create_draft(claire, request_type="annual_leave", start_date=future, end_date=future)
    app.leave.submit(claire, own.id, confirmed=True)
    with pytest.raises(PermissionDenied):
        app.leave.start_review(claire, own.id)
    with pytest.raises(PermissionDenied):
        app.leave.decide(claire, own.id, approve=True)


def test_approval_requires_the_review_step(app, claire, submitted):
    with pytest.raises(InvalidTransition):
        app.leave.decide(claire, submitted.id, approve=True, comment="trop vite")
    assert app.leave.get(claire, submitted.id).status == "submitted"


def test_employee_cannot_see_modify_or_cancel_a_colleagues_request(app, alice, bob, submitted):
    with pytest.raises(LeaveNotFound):
        app.leave.get(bob, submitted.id)
    with pytest.raises(LeaveNotFound):
        app.leave.cancel(bob, submitted.id)
    with pytest.raises(LeaveNotFound):
        app.leave.update_draft(bob, submitted.id, reason="piraté")
    assert app.leave.list_for_user(bob) == []
    with pytest.raises(PermissionDenied):
        app.leave.list_all(bob)


def test_someone_else_cannot_submit_a_draft(app, alice, bob, future):
    draft = app.leave.create_draft(alice, request_type="annual_leave", start_date=future, end_date=future)
    with pytest.raises(LeaveNotFound):
        app.leave.submit(bob, draft.id, confirmed=True)
    assert app.leave.get(alice, draft.id).status == "draft"


def test_hr_manager_cannot_submit_or_edit_someone_elses_draft(app, alice, claire, future):
    draft = app.leave.create_draft(alice, request_type="annual_leave", start_date=future, end_date=future)
    with pytest.raises(PermissionDenied):
        app.leave.submit(claire, draft.id, confirmed=True)
    with pytest.raises(PermissionDenied):
        app.leave.update_draft(claire, draft.id, reason="modifié par RH")


def test_submission_without_confirmation_is_refused(app, alice, future):
    draft = app.leave.create_draft(alice, request_type="annual_leave", start_date=future, end_date=future)
    with pytest.raises(ConfirmationRequired):
        app.leave.submit(alice, draft.id, confirmed=False)


def test_final_states_are_final(app, alice, claire, submitted):
    app.leave.start_review(claire, submitted.id)
    app.leave.decide(claire, submitted.id, approve=False, comment="Période de clôture")
    for call in (
        lambda: app.leave.decide(claire, submitted.id, approve=True, comment="finalement oui"),
        lambda: app.leave.cancel(alice, submitted.id),
        lambda: app.leave.submit(alice, submitted.id, confirmed=True),
    ):
        with pytest.raises(InvalidTransition):
            call()
    assert app.leave.get(alice, submitted.id).status == "rejected"


def test_status_cannot_be_written_through_update_fields(app, alice, future):
    draft = app.leave.create_draft(alice, request_type="annual_leave", start_date=future, end_date=future)
    app.leave.update_draft(alice, draft.id, status="approved", decided_by=alice.id, reason="tentative")  # extra keys ignored
    after = app.leave.get(alice, draft.id)
    assert after.status == "draft" and after.decided_by is None


def test_sql_metacharacters_in_user_input_are_data(app, alice, future):
    evil = "x'); DROP TABLE leave_requests; --"
    draft = app.leave.create_draft(alice, request_type="annual_leave", start_date=future, end_date=future, reason=evil)
    assert app.leave.get(alice, draft.id).reason == evil
    assert app.db.query_one("SELECT count(*) AS n FROM leave_requests")["n"] == 1


def test_validation_cannot_be_bypassed_by_the_service_layer(app, alice):
    with pytest.raises(LeaveValidationError):
        app.leave.create_draft(alice, request_type="annual_leave", start_date="2026-12-31", end_date="2026-01-01")


def test_escalations_are_private_to_their_author_and_hr(app, alice, bob, claire):
    draft = app.escalations.prepare(alice, "Question confidentielle sur mon contrat ?", None)
    saved = app.escalations.submit(alice, draft, confirmed=True)
    with pytest.raises(EscalationNotFound):
        app.escalations.get(bob, saved.id)
    assert app.escalations.list_for_user(bob) == []
    with pytest.raises(PermissionDenied):
        app.escalations.list_all(bob)
    with pytest.raises(PermissionDenied):
        app.escalations.resolve(alice, saved.id, "je me réponds moi-même")
    assert app.escalations.get(claire, saved.id).id == saved.id


def test_service_functions_refuse_unauthenticated_callers(app):
    with pytest.raises(PermissionDenied):
        app.engine.answer(None, "Bonjour ?")
    with pytest.raises(PermissionDenied):
        app.documents.list_documents(None)
    with pytest.raises(PermissionDenied):
        LeaveService(app.db).prepare(None, {})


def test_demo_passwords_are_stored_hashed(app):
    for row in app.db.query("SELECT password_hash FROM users"):
        assert row["password_hash"].startswith("pbkdf2_sha256$")
        assert "test-pw" not in row["password_hash"]
    assert auth.get_user_by_username(app.db, "ALICE") is not None  # usernames are case-insensitive

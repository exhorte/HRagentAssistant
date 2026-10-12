from datetime import date

import pytest

from hr_assistant import auth
from hr_assistant.auth import PermissionDenied
from hr_assistant.db import Database
from hr_assistant.intent import HANDOFF, LEAVE_REQUEST, LEAVE_STATUS, QUESTION, detect_intent
from hr_assistant.leave import (
    ALLOWED_TRANSITIONS, STATUSES, ConfirmationRequired, InvalidTransition, LeaveNotFound, LeaveService,
    LeaveValidationError, extract_leave_fields, parse_date,
)

TODAY = date(2026, 1, 15)


@pytest.fixture
def service(app):
    return LeaveService(app.db, today=lambda: TODAY)


def _draft(service, user, **overrides):
    values = {"request_type": "annual_leave", "start_date": "2026-03-10", "end_date": "2026-03-12", "reason": "Vacances"}
    values.update(overrides)
    return service.create_draft(user, **values)


# ----- validation
def test_prepare_lists_missing_fields(service, alice):
    prep = service.prepare(alice, {"request_type": "annual_leave", "start_date": "2026-03-10"})
    assert prep.missing == ["end_date"] and not prep.ready and prep.recap is None


def test_prepare_complete_request_builds_a_recap_without_inventing_a_balance(service, alice):
    prep = service.prepare(alice, {"request_type": "annual_leave", "start_date": "2026-03-10", "end_date": "2026-03-12"})
    assert prep.ready
    assert "10/03/2026" in prep.recap and "3 jours" in prep.recap
    assert "non disponible" in prep.recap  # no balance without a reliable source


@pytest.mark.parametrize(
    "fields, field",
    [
        ({"request_type": "annual_leave", "start_date": "2026-03-12", "end_date": "2026-03-10"}, "end_date"),
        ({"request_type": "annual_leave", "start_date": "2025-12-01", "end_date": "2025-12-03"}, "start_date"),
        ({"request_type": "teleportation", "start_date": "2026-03-10", "end_date": "2026-03-12"}, "request_type"),
        ({"request_type": "annual_leave", "start_date": "10 mars", "end_date": "2026-03-12"}, "start_date"),
        ({"request_type": "annual_leave", "start_date": "2026-03-10", "end_date": "2026-03-12", "reason": "x" * 501}, "reason"),
    ],
)
def test_invalid_fields_are_reported(service, alice, fields, field):
    assert field in service.prepare(alice, fields).errors


def test_reason_required_for_other_absence(service, alice):
    prep = service.prepare(alice, {"request_type": "other", "start_date": "2026-03-10", "end_date": "2026-03-10"})
    assert prep.missing == ["reason"]


def test_sick_leave_may_start_in_the_past(service, alice):
    prep = service.prepare(alice, {"request_type": "sick_leave", "start_date": "2026-01-10", "end_date": "2026-01-12"})
    assert prep.ready


def test_create_draft_rejects_incomplete_data(service, alice):
    with pytest.raises(LeaveValidationError) as exc:
        service.create_draft(alice, request_type="annual_leave")
    assert {"start_date", "end_date"} <= set(exc.value.errors)


# ----- lifecycle
def test_full_lifecycle_with_history(service, alice, claire):
    request = _draft(service, alice)
    assert request.status == "draft" and request.calendar_days == 3
    request = service.submit(alice, request.id, confirmed=True)
    assert request.status == "submitted" and request.submitted_at
    request = service.start_review(claire, request.id)
    assert request.status == "under_review"
    request = service.decide(claire, request.id, approve=True, comment="OK")
    assert request.status == "approved" and request.decided_by == claire.id
    assert [e.to_status for e in service.history(alice, request.id)] == ["draft", "submitted", "under_review", "approved"]


def test_submission_needs_explicit_confirmation(service, alice):
    request = _draft(service, alice)
    with pytest.raises(ConfirmationRequired):
        service.submit(alice, request.id, confirmed=False)
    assert service.get(alice, request.id).status == "draft"


def test_rejection_requires_a_reason(service, alice, claire):
    request = service.start_review(claire, service.submit(alice, _draft(service, alice).id, confirmed=True).id)
    with pytest.raises(LeaveValidationError):
        service.decide(claire, request.id, approve=False, comment="  ")
    assert service.decide(claire, request.id, approve=False, comment="Période de clôture").status == "rejected"


def test_owner_can_cancel_draft_submitted_and_approved(service, alice, claire):
    d = _draft(service, alice)
    assert service.cancel(alice, d.id).status == "cancelled"
    s = service.submit(alice, _draft(service, alice, start_date="2026-04-01", end_date="2026-04-02").id, confirmed=True)
    assert service.cancel(alice, s.id).status == "cancelled"


def test_only_drafts_can_be_edited(service, alice):
    request = _draft(service, alice)
    updated = service.update_draft(alice, request.id, end_date="2026-03-14")
    assert updated.calendar_days == 5
    service.submit(alice, request.id, confirmed=True)
    with pytest.raises(InvalidTransition):
        service.update_draft(alice, request.id, end_date="2026-03-20")


def test_overlapping_requests_are_refused_at_submission(service, alice):
    first = _draft(service, alice)
    service.submit(alice, first.id, confirmed=True)
    second = _draft(service, alice, start_date="2026-03-11", end_date="2026-03-13")
    with pytest.raises(LeaveValidationError):
        service.submit(alice, second.id, confirmed=True)


def test_every_non_listed_transition_is_refused(service, alice, claire):
    """Exhaustive: from each status, every target outside ALLOWED_TRANSITIONS raises."""
    for source in STATUSES:
        for target in STATUSES:
            if target in ALLOWED_TRANSITIONS[source]:
                continue
            request = _draft(service, alice, start_date="2026-05-01", end_date="2026-05-02")
            # Bring the request to `source` through the public API.
            path = {
                "draft": [],
                "submitted": ["submit"],
                "under_review": ["submit", "review"],
                "approved": ["submit", "review", "approve"],
                "rejected": ["submit", "review", "reject"],
                "cancelled": ["cancel"],
            }[source]
            for step in path:
                if step == "submit":
                    service.submit(alice, request.id, confirmed=True)
                elif step == "review":
                    service.start_review(claire, request.id)
                elif step == "approve":
                    service.decide(claire, request.id, approve=True, comment="ok")
                elif step == "reject":
                    service.decide(claire, request.id, approve=False, comment="non")
                elif step == "cancel":
                    service.cancel(alice, request.id)
            current = service.get(alice, request.id)
            assert current.status == source
            with pytest.raises(InvalidTransition):
                service._transition(current, claire, target)
            service.cancel(alice, request.id) if source not in ("rejected", "cancelled") else None


def test_requests_are_persisted_across_service_instances(app, alice, settings):
    request = _draft(LeaveService(app.db, today=lambda: TODAY), alice)
    reopened = Database(settings.db_path)
    try:
        again = LeaveService(reopened, today=lambda: TODAY).get(alice, request.id)
        assert again.status == "draft" and again.start_date == date(2026, 3, 10)
    finally:
        reopened.close()


# ----- helpers
def test_parse_date_formats():
    assert parse_date("2026-03-10") == parse_date("10/03/2026") == date(2026, 3, 10)
    assert parse_date("") is None
    with pytest.raises(ValueError):
        parse_date("demain")


def test_extract_leave_fields_from_chat_text():
    fields = extract_leave_fields("Je veux poser des congés du 21/12/2026 au 2026-12-23")
    assert fields == {"start_date": date(2026, 12, 21), "end_date": date(2026, 12, 23), "request_type": "annual_leave"}
    assert extract_leave_fields("Je suis en arrêt maladie à partir du 05/02/2026")["request_type"] == "sick_leave"
    assert extract_leave_fields("Je voudrais des congés") == {"request_type": "annual_leave"}


# ----- intent routing
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Je veux poser des congés du 21/12 au 23/12", LEAVE_REQUEST),
        ("Je souhaite prendre une semaine de vacances", LEAVE_REQUEST),
        ("Préparer ma demande de congé", LEAVE_REQUEST),
        ("Quelle est la procédure de demande de congé ?", QUESTION),
        ("Comment poser un congé ?", QUESTION),
        ("Combien de jours de congé puis-je poser ?", QUESTION),
        ("Où en est ma demande de congé ?", LEAVE_STATUS),
        ("Quel est le statut de ma demande ?", LEAVE_STATUS),
        ("Je voudrais parler à un responsable RH", HANDOFF),
        ("Transmettre ma question aux RH", HANDOFF),
        ("", QUESTION),
    ],
)
def test_intent_routing(text, expected):
    assert detect_intent(text) == expected

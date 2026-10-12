"""Streamlit interface of HRagentAssistant (prototype).

Run with:  streamlit run app.py     (or: python main.py serve)

Three areas behind a login: the HR assistant (chat with sources), the HR
documents, and the HR requests (leave workflow + hand-offs to HR).

All rules - access control, status transitions, confirmations - are enforced by
the backend services in ``hr_assistant``; this file only displays and collects.
"""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from hr_assistant import auth, leave as leave_mod
from hr_assistant.answering import ABSTAINED, ANSWERED, BLOCKED, CONFLICT, ERROR, Answer
from hr_assistant.auth import PermissionDenied, User
from hr_assistant.config import Settings
from hr_assistant.document_loader import DocumentError
from hr_assistant.documents import STATUSES as DOCUMENT_STATUSES
from hr_assistant.escalation import NO_NOTIFICATION_NOTICE, STATUS_LABELS as ESCALATION_LABELS
from hr_assistant.leave import (
    ALLOWED_TRANSITIONS, ConfirmationRequired, InvalidTransition, LeaveError, LeaveValidationError, REQUEST_TYPES,
)
from hr_assistant.logger import configure_logging, get_logger
from hr_assistant.pipeline import HRAssistantApp, build_app

logger = get_logger("hr_assistant.ui")

st.set_page_config(page_title="HRagentAssistant", page_icon="🧑‍💼", layout="wide")

PAGE_ASSISTANT, PAGE_DOCUMENTS, PAGE_REQUESTS = "Assistant RH", "Documents RH", "Demandes RH"
GENERIC_ERROR = "Une erreur est survenue. Réessayez ; si le problème persiste, consultez les journaux techniques."


@st.cache_resource(show_spinner="Initialisation de l'assistant…")
def _build(cache_key: str) -> HRAssistantApp:  # noqa: ARG001 - the key only distinguishes configurations
    settings = Settings.from_env()
    configure_logging(settings.data_dir / "logs")
    return build_app(settings)


def get_app() -> HRAssistantApp:
    return _build(repr(Settings.from_env()))


def _fmt_date(value: str) -> str:
    return value[:10]


# Navigation changes are done in button callbacks: they run *before* the next script run, so the
# sidebar radio (key "nav") may be updated without touching an already-instantiated widget.
def _start_handoff(app: HRAssistantApp, user: User, question: str, answer: Answer | None) -> None:
    st.session_state.handoff_draft = app.escalations.prepare(user, question, answer)
    st.session_state.nav = PAGE_REQUESTS


def _open_leave_form(fields: dict) -> None:
    st.session_state.leave_prefill = fields
    st.session_state.nav = PAGE_REQUESTS


# --------------------------------------------------------------------------- login
def render_login(app: HRAssistantApp) -> None:
    st.title("🧑‍💼 HRagentAssistant")
    st.warning(
        "**Prototype local — données synthétiques uniquement.** Cette démonstration n'est pas prête à recevoir "
        "des données RH réelles (voir docs/SECURITY.md)."
    )
    with st.form("login"):
        username = st.text_input("Identifiant")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter")
    if submitted:
        user = app.login(username, password)
        if user is None:
            st.error("Identifiant ou mot de passe incorrect.")
        else:
            st.session_state.user = user
            st.session_state.messages = []
            st.rerun()
    st.caption(
        "Comptes de démonstration : créés par `python main.py init` ; les mots de passe générés sont dans "
        "`data/runtime/demo_credentials.txt` (fichier local, ignoré par git)."
    )


# --------------------------------------------------------------------------- page A: assistant
_STATUS_BANNERS = {
    ANSWERED: ("success", "Réponse fondée sur les documents"),
    ABSTAINED: ("info", "Pas de réponse fiable dans les documents disponibles"),
    CONFLICT: ("warning", "Informations contradictoires entre documents"),
    BLOCKED: ("error", "Demande refusée"),
    ERROR: ("error", "Service momentanément indisponible"),
}


_MODE_LABELS = {"extractive": "extractif", "llm": "génératif vérifié"}


def _as_markdown(text: str) -> str:
    """Turn the answer's ``•`` lines into a real Markdown list (single newlines would be merged)."""
    return "\n\n".join(f"- {line[2:]}" if line.startswith("• ") else line for line in text.split("\n") if line.strip())


def _render_answer(app: HRAssistantApp, user: User, answer: Answer, question: str, key: str) -> None:
    level, title = _STATUS_BANNERS.get(answer.status, ("info", answer.status))
    mode = _MODE_LABELS.get(answer.mode)
    getattr(st, level)(f"**{title}**" + (f" · mode {mode}" if mode else ""))
    st.markdown(_as_markdown(answer.text))
    for warning in answer.warnings:
        st.warning(warning)
    if answer.citations:
        with st.expander(f"Sources consultées ({len(answer.citations)})", expanded=answer.status != ANSWERED):
            for c in answer.citations:
                st.markdown(f"**[{c.marker}] {c.label()}**")
                st.caption(c.quote)
    if answer.suggest_handoff and answer.status != BLOCKED:
        st.button(
            "Transmettre cette question aux RH", key=f"handoff-{key}",
            on_click=_start_handoff, args=(app, user, question, answer),
        )


def page_assistant(app: HRAssistantApp, user: User) -> None:
    st.header("Assistant RH")
    provider = app.settings.llm_provider
    st.caption(
        "Mode **extractif** (aucun modèle de langage : phrases extraites des documents)."
        if provider == "none"
        else f"Mode **génératif** via `{provider}`, avec vérification des citations."
    )
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["text"])
            elif message.get("answer") is not None:
                _render_answer(app, user, message["answer"], message["question"], key=str(i))
            else:
                st.markdown(message["text"])
                if message.get("leave_fields") is not None:
                    st.button("Ouvrir la demande de congé", key=f"leave-{i}",
                              on_click=_open_leave_form, args=(message["leave_fields"],))
                if message.get("handoff"):
                    st.button("Préparer un transfert aux RH", key=f"hand-{i}",
                              on_click=_start_handoff, args=(app, user, message["question"], None))

    question = st.chat_input("Posez votre question sur les documents RH…")
    if question:
        st.session_state.messages.append({"role": "user", "text": question})
        try:
            reply = app.handle_message(user, question)
        except Exception:
            logger.exception("Assistant failure")
            st.session_state.messages.append({"role": "assistant", "text": GENERIC_ERROR})
        else:
            entry: dict = {"role": "assistant", "question": question}
            if reply.answer is not None:
                entry["answer"] = reply.answer
            else:
                entry["text"] = reply.message or ""
                if reply.intent == "leave_request":
                    entry["leave_fields"] = reply.leave_fields
                entry["handoff"] = reply.intent == "handoff"
            st.session_state.messages.append(entry)
        st.rerun()


# --------------------------------------------------------------------------- page B: documents
_INDEX_BADGES = {"indexed": "✅ indexé", "failed": "❌ échec", "pending": "⏳ en attente"}


def page_documents(app: HRAssistantApp, user: User) -> None:
    st.header("Documents RH")
    st.caption("Formats acceptés : PDF (texte), DOCX, TXT. Utilisez uniquement des documents synthétiques ou autorisés.")
    is_hr = user.can(auth.DOCUMENTS_MANAGE)

    if is_hr:
        with st.expander("Importer un document", expanded=False):
            with st.form("upload", clear_on_submit=True):
                file = st.file_uploader("Fichier", type=["pdf", "docx", "txt"])
                col1, col2, col3, col4 = st.columns(4)
                version = col1.text_input("Version", placeholder="1.0")
                family = col2.text_input("Famille (optionnel)", placeholder="politique-conges")
                status = col3.selectbox("Statut", DOCUMENT_STATUSES)
                access = col4.selectbox(
                    "Accès", [auth.ACCESS_ALL, auth.ACCESS_HR_ONLY],
                    format_func=lambda v: "Tous les salariés" if v == auth.ACCESS_ALL else "Responsables RH uniquement",
                )
                go = st.form_submit_button("Importer et indexer")
            if go:
                if file is None:
                    st.error("Choisissez un fichier.")
                else:
                    try:
                        record = app.documents.import_bytes(
                            user, file.name, file.getvalue(), version=version or None, family=family or None,
                            status=status, access_level=access,
                        )
                        st.success(f"« {record.name} » indexé : {record.chunk_count} passages.")
                    except DocumentError as exc:
                        st.error(str(exc))
                    except PermissionDenied:
                        st.error("Action non autorisée.")
                    except Exception:
                        logger.exception("Upload failure")
                        st.error(GENERIC_ERROR)

        if st.button("Réindexer tous les documents"):
            ok, failed = app.documents.reindex_all(user)
            st.info(f"Réindexation : {ok} réussi(s), {failed} en échec.")

    documents = app.documents.list_documents(user)
    if not documents:
        st.info("Aucun document importé. Lancez `python main.py init` ou importez un fichier.")
        return
    for doc in documents:
        stale = doc.indexing_status == "indexed" and doc.collection != app.index.collection
        with st.container(border=True):
            cols = st.columns([4, 2, 2, 2])
            cols[0].markdown(f"**{doc.name}**" + ("  · _démo_" if doc.is_demo else ""))
            cols[0].caption(
                f"{doc.doc_type.upper()} · version {doc.version or '—'} · langue {doc.language or '?'} · "
                f"importé le {_fmt_date(doc.imported_at)}"
            )
            cols[1].write(_INDEX_BADGES.get(doc.indexing_status, doc.indexing_status) + (" · à réindexer" if stale else ""))
            cols[2].write(f"Statut : {doc.status}")
            cols[3].write("🔒 RH uniquement" if doc.access_level == auth.ACCESS_HR_ONLY else "👥 Tous")
            if doc.indexing_status == "failed" and doc.indexing_error:
                st.error(doc.indexing_error)
            if is_hr:
                a, b, c, _ = st.columns([2, 2, 3, 3])
                if a.button("Réindexer", key=f"re-{doc.id}"):
                    try:
                        app.documents.reindex(user, doc.id)
                        st.rerun()
                    except DocumentError as exc:
                        st.error(str(exc))
                new_status = c.selectbox(
                    "Statut", DOCUMENT_STATUSES, index=DOCUMENT_STATUSES.index(doc.status),
                    key=f"st-{doc.id}", label_visibility="collapsed",
                )
                if new_status != doc.status:
                    try:
                        app.documents.update_metadata(user, doc.id, status=new_status)
                        st.rerun()
                    except DocumentError as exc:
                        st.error(str(exc))
                confirm = b.checkbox("Confirmer le retrait", key=f"cf-{doc.id}")
                if b.button("Retirer", key=f"rm-{doc.id}", disabled=not confirm):
                    app.documents.remove(user, doc.id)
                    st.rerun()


# --------------------------------------------------------------------------- page C: requests
def _render_leave_row(app: HRAssistantApp, user: User, request, *, hr_view: bool) -> None:
    with st.container(border=True):
        st.markdown(
            f"**{request.type_label}** · du {request.start_date:%d/%m/%Y} au {request.end_date:%d/%m/%Y} "
            f"({request.calendar_days} j) · **{request.status_label}**"
        )
        if request.reason:
            st.caption(f"Motif : {request.reason}")
        if request.decision_comment:
            st.caption(f"Commentaire RH : {request.decision_comment}")
        if hr_view:
            st.caption(f"Demandeur : {_username(app, request.user_id)}")
        with st.expander("Historique"):
            for event in app.leave.history(user, request.id):
                st.write(
                    f"{_fmt_date(event.at)} · {leave_mod.STATUS_LABELS.get(event.to_status, event.to_status)}"
                    + (f" — {event.comment}" if event.comment else "")
                )
        actions: list[tuple[str, str]] = []
        is_owner = request.user_id == user.id
        allowed = ALLOWED_TRANSITIONS[request.status]
        if is_owner and leave_mod.SUBMITTED in allowed:
            actions.append(("submit", "Confirmer et soumettre"))
        if is_owner and leave_mod.CANCELLED in allowed:
            actions.append(("cancel", "Annuler la demande"))
        if hr_view and not is_owner and user.can(auth.LEAVE_DECIDE):
            if leave_mod.UNDER_REVIEW in allowed:
                actions.append(("review", "Prendre en charge"))
            if request.status == leave_mod.UNDER_REVIEW:
                actions.append(("approve", "Approuver"))
                actions.append(("reject", "Refuser"))
        comment = ""
        if any(a == "reject" for a, _ in actions):
            comment = st.text_input("Motif du refus / commentaire", key=f"cm-{request.id}")
        if actions:
            cols = st.columns(len(actions))
            for col, (action, label) in zip(cols, actions):
                if col.button(label, key=f"{action}-{request.id}"):
                    _run_leave_action(app, user, request.id, action, comment)


def _username(app: HRAssistantApp, user_id: str) -> str:
    found = auth.get_user(app.db, user_id)
    return found.display_name if found else "?"


def _run_leave_action(app: HRAssistantApp, user: User, request_id: str, action: str, comment: str) -> None:
    try:
        if action == "submit":
            app.leave.submit(user, request_id, confirmed=True)
        elif action == "cancel":
            app.leave.cancel(user, request_id)
        elif action == "review":
            app.leave.start_review(user, request_id)
        elif action == "approve":
            app.leave.decide(user, request_id, approve=True, comment=comment)
        elif action == "reject":
            app.leave.decide(user, request_id, approve=False, comment=comment)
        st.rerun()
    except (LeaveValidationError, InvalidTransition, ConfirmationRequired, LeaveError) as exc:
        st.error(str(exc))
    except PermissionDenied as exc:
        st.error(str(exc))


def _leave_form(app: HRAssistantApp, user: User) -> None:
    prefill = st.session_state.get("leave_prefill") or {}
    candidate = st.session_state.get("leave_candidate")
    if candidate is None:
        with st.form("leave-form"):
            types = list(REQUEST_TYPES)
            request_type = st.selectbox(
                "Type de demande", types, index=types.index(prefill.get("request_type", "annual_leave")),
                format_func=lambda v: REQUEST_TYPES[v],
            )
            col1, col2 = st.columns(2)
            start = col1.date_input("Date de début", value=prefill.get("start_date") or date.today() + timedelta(days=14))
            end = col2.date_input("Date de fin", value=prefill.get("end_date") or date.today() + timedelta(days=14))
            reason = st.text_area("Motif (facultatif sauf « Autre absence »)", max_chars=leave_mod.MAX_REASON_CHARS)
            go = st.form_submit_button("Vérifier et afficher le récapitulatif")
        if go:
            fields = {"request_type": request_type, "start_date": start, "end_date": end, "reason": reason}
            prep = app.leave.prepare(user, fields)
            if prep.ready:
                st.session_state.leave_candidate = fields
                st.session_state.pop("leave_prefill", None)
                st.rerun()
            for name in prep.missing:
                st.error(f"Information manquante : {leave_mod.FIELD_LABELS.get(name, name)}.")
            for name, message in prep.errors.items():
                st.error(f"{leave_mod.FIELD_LABELS.get(name, name)} : {message}")
        return

    prep = app.leave.prepare(user, candidate)
    st.subheader("Récapitulatif")
    st.text(prep.recap or "")
    confirmed = st.checkbox("Je confirme l'exactitude de ces informations", key="leave-confirm")
    col1, col2, col3 = st.columns(3)
    try:
        if col1.button("Confirmer et soumettre", disabled=not confirmed):
            draft = app.leave.create_draft(user, **candidate)
            app.leave.submit(user, draft.id, confirmed=confirmed)
            st.session_state.pop("leave_candidate", None)
            st.success("Demande soumise. Elle sera examinée par les RH ; son statut est visible dans « Mes demandes ».")
        if col2.button("Enregistrer en brouillon"):
            app.leave.create_draft(user, **candidate)
            st.session_state.pop("leave_candidate", None)
            st.success("Brouillon enregistré (non soumis).")
        if col3.button("Modifier"):
            st.session_state.pop("leave_candidate", None)
            st.rerun()
    except (LeaveError, PermissionDenied) as exc:
        st.error(str(exc))


def _render_handoff(app: HRAssistantApp, user: User) -> None:
    draft = st.session_state.get("handoff_draft")
    if draft is None:
        return
    st.subheader("Transfert à un responsable RH")
    st.info(draft.reason)
    st.text(draft.summary)
    note = st.text_input("Précision à ajouter (facultatif)", key="handoff-note")
    st.caption(NO_NOTIFICATION_NOTICE)
    c1, c2 = st.columns(2)
    if c1.button("Enregistrer le transfert"):
        saved = app.escalations.submit(user, draft.with_note(note), confirmed=True)
        st.session_state.pop("handoff_draft", None)
        st.success(f"Transfert n° {saved.id[:8]} enregistré (statut : {saved.status_label}).")
    if c2.button("Annuler le transfert"):
        st.session_state.pop("handoff_draft", None)
        st.rerun()


def page_requests(app: HRAssistantApp, user: User) -> None:
    st.header("Demandes RH")
    st.caption(f"{leave_mod.BALANCE_NOTICE}")
    _render_handoff(app, user)
    is_hr = user.can(auth.LEAVE_DECIDE)
    labels = ["Nouvelle demande de congé", "Mes demandes", "Mes transferts RH"]
    if is_hr:
        labels += ["À traiter (RH)", "Transferts reçus (RH)"]
    tabs = st.tabs(labels)

    with tabs[0]:
        _leave_form(app, user)
    with tabs[1]:
        requests = app.leave.list_for_user(user)
        if not requests:
            st.info("Aucune demande pour le moment.")
        for request in requests:
            _render_leave_row(app, user, request, hr_view=False)
    with tabs[2]:
        items = app.escalations.list_for_user(user)
        if not items:
            st.info("Aucun transfert.")
        for item in items:
            with st.container(border=True):
                st.markdown(f"**{item.status_label}** · {_fmt_date(item.created_at)}")
                st.text(item.summary)
                if item.resolution_note:
                    st.success(f"Réponse des RH : {item.resolution_note}")
    if is_hr:
        with tabs[3]:
            pending = [r for r in app.leave.list_all(user) if r.status != leave_mod.DRAFT]
            if not pending:
                st.info("Aucune demande à traiter.")
            for request in pending:
                _render_leave_row(app, user, request, hr_view=True)
        with tabs[4]:
            items = app.escalations.list_all(user)
            if not items:
                st.info("Aucun transfert reçu.")
            for item in items:
                with st.container(border=True):
                    st.markdown(f"**{item.status_label}** · {_fmt_date(item.created_at)} · {_username(app, item.user_id)}")
                    st.text(item.summary)
                    if item.status == "open" and st.button("Prendre en charge", key=f"take-{item.id}"):
                        app.escalations.take(user, item.id)
                        st.rerun()
                    if item.status != "resolved":
                        note = st.text_input("Note de résolution", key=f"note-{item.id}")
                        if st.button("Marquer comme résolu", key=f"res-{item.id}"):
                            try:
                                app.escalations.resolve(user, item.id, note)
                                st.rerun()
                            except ValueError as exc:
                                st.error(str(exc))


# --------------------------------------------------------------------------- main
def main() -> None:
    app = get_app()
    user: User | None = st.session_state.get("user")
    if user is None:
        render_login(app)
        return

    st.sidebar.title("🧑‍💼 HRagentAssistant")
    st.sidebar.write(f"**{user.display_name}**")
    st.sidebar.caption("Rôle : responsable RH" if user.role == auth.HR_MANAGER else "Rôle : employé")
    pages = [PAGE_ASSISTANT, PAGE_DOCUMENTS, PAGE_REQUESTS]
    if st.session_state.get("nav") not in pages:
        st.session_state.nav = PAGE_ASSISTANT
    choice = st.sidebar.radio("Navigation", pages, key="nav")
    if st.sidebar.button("Se déconnecter"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.sidebar.divider()
    st.sidebar.caption("Prototype · données synthétiques uniquement · aucune notification n'est envoyée.")

    try:
        {PAGE_ASSISTANT: page_assistant, PAGE_DOCUMENTS: page_documents, PAGE_REQUESTS: page_requests}[choice](app, user)
    except PermissionDenied:
        st.error("Action non autorisée.")
    except Exception:
        logger.exception("Unhandled UI error")
        st.error(GENERIC_ERROR)


main()

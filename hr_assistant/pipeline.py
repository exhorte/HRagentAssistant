"""Wires all the components together into one ready-to-use application object.

This is the single entry point that ``main.py`` (CLI), ``app.py`` (Streamlit)
and the tests all call. Each concern lives in its own small module; this file
only composes them:

    config → db → embeddings → vector index → documents / answer engine / leave / escalation
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hr_assistant import auth, intent
from hr_assistant.answering import Answer, AnswerEngine
from hr_assistant.auth import User
from hr_assistant.config import Settings
from hr_assistant.db import Database
from hr_assistant.documents import DocumentService
from hr_assistant.escalation import EscalationService
from hr_assistant.guardrails import LLMGuard
from hr_assistant.leave import LeaveService, extract_leave_fields
from hr_assistant.llm import get_llm
from hr_assistant.logger import get_logger
from hr_assistant.vector_store import KnowledgeIndex

logger = get_logger(__name__)

_UNSET = object()


@dataclass
class ChatReply:
    """Outcome of one chat message: either an ``answer`` (documents) or a workflow ``message``."""

    intent: str
    answer: Answer | None = None
    message: str | None = None
    leave_fields: dict = field(default_factory=dict)  # pre-filled values for the leave form
    leave_missing: list[str] = field(default_factory=list)


class HRAssistantApp:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        index: KnowledgeIndex,
        documents: DocumentService,
        engine: AnswerEngine,
        leave: LeaveService,
        escalations: EscalationService,
    ):
        self.settings = settings
        self.db = db
        self.index = index
        self.documents = documents
        self.engine = engine
        self.leave = leave
        self.escalations = escalations

    # ----- authentication -----
    def login(self, username: str, password: str) -> User | None:
        return auth.authenticate(self.db, username, password)

    # ----- chat -----
    def handle_message(self, user: User, text: str) -> ChatReply:
        """Route a chat message. Questions always go through the verified RAG pipeline."""
        kind = intent.detect_intent(text)
        if kind == intent.LEAVE_REQUEST:
            fields = extract_leave_fields(text)
            prep = self.leave.prepare(user, fields)
            if not prep.ready:
                return ChatReply(
                    kind,
                    message=(
                        "Je peux vous aider à préparer une demande de congé. "
                        "Complétez-la dans l'onglet « Demandes RH » : rien n'est enregistré tant que vous n'avez pas confirmé. "
                        "Je ne peux pas vous indiquer de solde de congés (aucun système RH connecté)."
                    ),
                    leave_fields=fields,
                    leave_missing=list(prep.missing),
                )
            return ChatReply(
                kind,
                message="Voici le récapitulatif de la demande. Confirmez-la dans l'onglet « Demandes RH » :\n" + (prep.recap or ""),
                leave_fields=fields,
            )
        if kind == intent.LEAVE_STATUS:
            return ChatReply(kind, message="Vous pouvez consulter le statut de vos demandes dans l'onglet « Demandes RH ».")
        if kind == intent.HANDOFF:
            return ChatReply(
                kind,
                message="Je peux préparer un transfert vers un responsable RH : vous verrez un résumé à confirmer avant l'enregistrement.",
            )
        return ChatReply(intent.QUESTION, answer=self.engine.answer(user, text))

    def close(self) -> None:
        try:
            self.index.close()
        finally:
            self.db.close()


def build_app(
    settings: Settings | None = None,
    *,
    llm=_UNSET,
    embeddings=None,
    db: Database | None = None,
) -> HRAssistantApp:
    """Compose the application.

    ``llm`` may be injected (tests pass a scripted fake or ``None`` for extractive mode); by default it
    is built from ``settings.llm_provider``. ``embeddings`` and ``db`` can be injected too.
    """
    settings = settings or Settings.from_env()
    settings.validate()
    database = db or Database(settings.db_path)
    index = KnowledgeIndex.from_settings(settings, embeddings)
    model = get_llm(settings) if llm is _UNSET else llm
    guard = LLMGuard.from_settings(settings)
    app = HRAssistantApp(
        settings=settings,
        db=database,
        index=index,
        documents=DocumentService(database, index, settings),
        engine=AnswerEngine(index, settings, llm=model, llm_guard=guard),
        leave=LeaveService(database),
        escalations=EscalationService(database),
    )
    logger.info(
        "App ready: llm=%s embeddings=%s qdrant=%s collection=%s",
        settings.llm_provider, settings.embedding_backend, settings.qdrant_mode, settings.collection_name,
    )
    return app

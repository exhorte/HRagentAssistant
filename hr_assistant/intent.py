"""Rule-based routing of a chat message to a capability.

Deliberately not an LLM call: routing decides which *backend action* is
offered, so it must be deterministic and testable. Informational questions
that merely contain "demande de congé" ("Quelle est la procédure de demande
de congé ?") stay questions; only first-person requests start the workflow.
"""

from __future__ import annotations

import re

from hr_assistant.textutils import strip_accents

QUESTION = "question"
LEAVE_REQUEST = "leave_request"
LEAVE_STATUS = "leave_status"
HANDOFF = "handoff"

_INTERROGATIVE_START = re.compile(
    r"^(comment|combien|quel|quelle|quels|quelles|est-ce|est ce|pourquoi|ou |qui |quand|que |qu'|puis-je|peut-on|"
    r"what|how|when|where|why|who|which|can i|do i|is there)\b"
)
_FIRST_PERSON_LEAVE = re.compile(
    r"\bje\s+(?:veux|voudrais|souhaite|souhaiterais|aimerais|dois|vais|compte|desire)\s+(?:\w+\s+){0,3}?"
    r"(?:poser|prendre|demander|deposer|faire|preparer|creer|soumettre|partir)\s+(?:\w+\s+){0,5}?"
    r"(?:conges?|vacances|absence|jours? de repos|rtt)"
)
_IMPERATIVE_LEAVE = re.compile(
    r"\b(?:preparer|creer|deposer|soumettre|faire|rediger|lancer|ouvrir)\s+(?:\w+\s+){0,2}?demande\s+(?:de|d')\s*"
    r"(?:conges?|absence|vacances)"
)
_STATUS = re.compile(
    r"\b(?:statut|etat|avancement|suivi|ou en est|suivre)\b.{0,40}\b(?:demande|conges?)\b"
    r"|\b(?:ma|mes)\s+demandes?\b.{0,40}\b(?:valide|approuve|acceptee?|refusee?|traitee?|en cours)"
)
_HANDOFF = re.compile(
    r"\b(?:parler|contacter|joindre|ecrire|transmettre|transferer|escalader|voir|rencontrer)\s+(?:\w+\s+){0,4}?"
    r"(?:rh|ressources humaines|responsable rh|gestionnaire rh|humain|une personne)\b"
    r"|\b(?:transmettre|transferer)\s+(?:\w+\s+){0,3}?(?:rh|ressources humaines)\b"
)


def detect_intent(text: str) -> str:
    """Return one of ``leave_request``, ``leave_status``, ``handoff`` or ``question``."""
    folded = strip_accents(text or "").lower().replace("’", "'").strip()
    if not folded:
        return QUESTION
    if _FIRST_PERSON_LEAVE.search(folded):
        return LEAVE_REQUEST
    if _STATUS.search(folded):
        return LEAVE_STATUS
    if _HANDOFF.search(folded):
        return HANDOFF
    if _INTERROGATIVE_START.search(folded):
        return QUESTION
    if _IMPERATIVE_LEAVE.search(folded):
        return LEAVE_REQUEST
    return QUESTION

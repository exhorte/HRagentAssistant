"""Small, dependency-free text helpers shared by embeddings, retrieval and loading.

Normalisation is deliberately simple and deterministic (lower-case, accents
stripped, plural 's'/'x' removed, truncated to a 5-char stem) so that
"congé", "congés" and "CONGES" all map to the same term. It is a crude
stemmer, good enough for French/English HR vocabulary and for measuring
*lexical coverage* of a question by retrieved passages.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STEM_LEN = 5

_STOPWORDS_FR = """
a au aux avec ce ces cet cette ceci cela ca car chez comme comment combien dans de des du donc dont elle elles en
entre es est et etre ete etais etait sont suis sommes etes eu y il ils je la le les leur leurs lui ma mais me mes
moi mon ne ni nos notre nous on ou par pas pour pourquoi qu que quel quelle quelles quels qui quoi quand sa sans se
ses si son sur ta te tes toi ton tu un une vos votre vous plus tres peut peuvent peux doit dois doivent faut faire
fait avoir ai as avons avez ont aura sous vers aussi alors apres avant bien ainsi autre autres tout tous toute
toutes quelques quelque chaque meme dit dire etc afin lors lorsque puis selon cest jai
""".split()

_STOPWORDS_EN = """
a and are as at be been but by can could do does did for from has have how i if in into is it its may me my
no not of on or our shall should so than that the their them then there these they this those to up us was we
were what when where which who whom why will with would you your many much per each any
""".split()

# Words that fill questions but carry no topic ("quelles règles sont prévues dans notre entreprise ?").
_FILLERS = """
prevu prevus prevue prevues interne internes fonctionne fonctionnent necessaire necessaires possible possibles
savoir concernant propos existe existent exister expliquer explique expliquez indiquer indique indiquez
entreprise societe svp merci bonjour bonsoir obtenir prendre poser passe passer chose besoin cas pendant durant
jusqu jusque etape etapes droit droits regle regles modalite modalites
""".split()

STOPWORDS = frozenset(_STOPWORDS_FR) | frozenset(_STOPWORDS_EN) | frozenset(_FILLERS)

# Tiny HR thesaurus (stem -> canonical stem) so that "employé" matches "salarié", "processus" matches
# "procédure", etc. Applied after stemming; deliberately short and domain-specific.
_SYNONYMS = {
    "emplo": "salar",   # employé -> salarié
    "colla": "salar",   # collaborateur -> salarié
    "demar": "proce",   # démarche -> procédure (processus -> proce naturally)
    "vacan": "conge",   # vacances -> congé
    "remun": "salai",   # rémunération -> salaire
    "paie": "salai",    # paie -> salaire
    "piece": "docum",   # pièce -> document
    "an": "annee",      # "par an" -> "par année"
    "ans": "annee",
    "temp": "delai",    # "combien de temps" -> délai
    "manag": "respo",   # manager -> responsable
    "chef": "respo",
}

# Words typical for each language, used for a cheap language guess.
_FR_MARKERS = frozenset("le la les des du de un une et est sont pour par dans sur avec ou au aux votre vos nous vous que qui".split())
_EN_MARKERS = frozenset("the and is are for of to in on with or your you we that which be by an".split())


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    """Lower-case, accent-free alphanumeric tokens (no stop-word removal, no stemming)."""
    return _TOKEN_RE.findall(strip_accents(text).lower().replace("’", "'"))


def stem(token: str) -> str:
    if token.isdigit():
        return token
    if len(token) > 4 and token[-1] in "sx":
        token = token[:-1]
    short = token[:_STEM_LEN]
    return _SYNONYMS.get(short, short)


def content_terms(text: str) -> list[str]:
    """Stemmed non-stop-word terms in order of appearance (duplicates kept)."""
    return [stem(tok) for tok in tokenize(text) if tok not in STOPWORDS and len(tok) > 1]


def term_set(text: str) -> set[str]:
    return set(content_terms(text))


def detect_language(text: str) -> str:
    """Return 'fr', 'en' or 'unknown' from stop-word frequencies."""
    tokens = tokenize(text[:5000])
    if len(tokens) < 8:
        return "unknown"
    fr = sum(1 for t in tokens if t in _FR_MARKERS)
    en = sum(1 for t in tokens if t in _EN_MARKERS)
    if fr == 0 and en == 0:
        return "unknown"
    if fr >= en * 1.2:
        return "fr"
    if en >= fr * 1.2:
        return "en"
    return "unknown"


_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def numbers_in(text: str) -> set[str]:
    """Numeric tokens in ``text`` with a normalised decimal separator ('2,5' -> '2.5')."""
    return {m.replace(",", ".") for m in _NUMBER_RE.findall(text)}

from hr_assistant.textutils import (
    content_terms, detect_language, numbers_in, stem, strip_accents, term_set, tokenize,
)


def test_strip_accents_and_tokenize():
    assert strip_accents("Congé à l'étranger") == "Conge a l'etranger"
    assert tokenize("Télétravail : 2 jours/semaine !") == ["teletravail", "2", "jours", "semaine"]


def test_inflections_collapse_to_the_same_term():
    assert term_set("congé") == term_set("CONGÉS") == term_set("conges")
    assert stem("demandes") == stem("demande")


def test_stopwords_and_fillers_are_dropped():
    assert content_terms("Quelle est la procédure de demande ?") == ["proce", "deman"]
    # "prévus", "interne" are question filler, not topics
    assert "prevu" not in content_terms("les congés prévus en interne")


def test_hr_thesaurus_maps_synonyms():
    assert term_set("employés") == term_set("salariés")
    assert term_set("processus") == term_set("procédure")
    assert term_set("vacances") == term_set("congés")
    assert term_set("par an") == term_set("par année")


def test_unrelated_words_do_not_collide():
    assert term_set("salaire") != term_set("salarié")


def test_numbers_are_kept_as_terms_and_normalised():
    assert "25" in content_terms("25 jours")
    assert numbers_in("2,5 jours et 10 euros") == {"2.5", "10"}


def test_detect_language():
    assert detect_language("Les salariés peuvent poser des congés après accord de leur responsable pour la période.") == "fr"
    assert detect_language("All employees may work from home up to two days per week with the approval of their manager.") == "en"
    assert detect_language("12 34") == "unknown"

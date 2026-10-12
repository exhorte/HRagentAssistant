from hr_assistant.answering import _split_sentences, parse_model_json, verify_generated_answer

PASSAGES = {
    "S1": "Chaque salarié acquiert 25 jours ouvrés de congés payés par année.",
    "S2": "Un maximum de 5 jours peut être reporté. Indemnité de 2,5 euros.",
}


# ----- model JSON parsing
def test_parse_model_json_accepts_fenced_or_chatty_output():
    assert parse_model_json('{"answer": "ok", "citations": ["S1"]}')["answer"] == "ok"
    assert parse_model_json('Voici :\n```json\n{"answer": "ok"}\n```')["answer"] == "ok"


def test_parse_model_json_rejects_garbage():
    assert parse_model_json("pas de json") is None
    assert parse_model_json("{cassé") is None
    assert parse_model_json("[1, 2, 3]") is None


# ----- citation / number verification
def test_valid_answer_is_accepted():
    result = verify_generated_answer("Vous avez 25 jours de congés [S1].", ["S1"], PASSAGES)
    assert result.ok and result.used_markers == ["S1"]


def test_unknown_marker_is_rejected():
    result = verify_generated_answer("Vous avez 25 jours [S9].", ["S9"], PASSAGES)
    assert not result.ok
    assert any(p.startswith("invalid_citation:S9") for p in result.problems)


def test_answer_without_any_citation_is_rejected():
    result = verify_generated_answer("Vous avez 25 jours de congés.", [], PASSAGES)
    assert not result.ok and "no_citation" in result.problems


def test_number_absent_from_cited_passages_is_rejected():
    result = verify_generated_answer("Vous avez 30 jours de congés [S1].", ["S1"], PASSAGES)
    assert not result.ok and any(p.startswith("unsupported_numbers:30") for p in result.problems)


def test_number_must_come_from_a_cited_passage_not_just_any_passage():
    # 5 appears in S2 only; the answer cites S1.
    result = verify_generated_answer("Vous pouvez reporter 5 jours [S1].", ["S1"], PASSAGES)
    assert not result.ok


def test_decimal_comma_is_normalised():
    assert verify_generated_answer("Indemnité de 2,5 euros [S2].", ["S2"], PASSAGES).ok


def test_citation_found_in_text_even_if_not_declared():
    assert verify_generated_answer("Vous avez 25 jours [S1].", [], PASSAGES).ok


# ----- extractive sentence splitting
def test_wrapped_pdf_lines_are_rejoined():
    text = "Le délai est de 5 jours ouvrés pour une attestation\nde travail et de 10 jours pour une attestation de salaire."
    assert _split_sentences(text) == [text.replace("\n", " ")]


def test_sentences_and_bullets_are_split():
    text = "Mariage : 4 jours. Naissance : 3 jours.\n- Décès : 3 jours."
    assert _split_sentences(text) == ["Mariage : 4 jours.", "Naissance : 3 jours.", "Décès : 3 jours."]

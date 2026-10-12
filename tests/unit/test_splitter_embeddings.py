import math

from hr_assistant.document_loader import ExtractedDocument, Page
from hr_assistant.embeddings import HASHING_DIM, HashingEmbeddings
from hr_assistant.splitter import split_into_chunks

META = {"document_id": "doc-1", "document_name": "politique_conges.txt", "document_type": "txt", "tenant_id": "default"}


def _doc(*pages, doc_type="txt"):
    return ExtractedDocument(doc_type=doc_type, pages=[Page(n, t) for n, t in pages])


# ----- splitter
def test_sections_become_chunks_with_title_and_section_header():
    extracted = _doc((None, "## POLITIQUE DE CONGÉS\nVersion 2.0.\n## 1. Congés payés\n25 jours par an.\n## 2. Report\n5 jours."))
    chunks = split_into_chunks(extracted, META)
    assert [c.metadata["section"] for c in chunks] == ["POLITIQUE DE CONGÉS", "1. Congés payés", "2. Report"]
    assert chunks[1].page_content.startswith("POLITIQUE DE CONGÉS\n1. Congés payés\n25 jours par an.")
    assert all(c.metadata["document_id"] == "doc-1" for c in chunks)
    assert len({c.metadata["chunk_id"] for c in chunks}) == 3


def test_all_caps_lines_are_headings_too():
    extracted = _doc((None, "COMPANY HR POLICY\n1. LEAVE POLICY\nTwenty days per year.\n2. REMOTE WORK\nTwo days."))
    assert [c.metadata["section"] for c in split_into_chunks(extracted, META)][-2:] == ["1. LEAVE POLICY", "2. REMOTE WORK"]


def test_pdf_page_numbers_and_heading_carry_over_pages():
    extracted = _doc((1, "TITRE\n1. DELAIS\nCinq jours."), (2, "suite du texte sur la page deux."), doc_type="pdf")
    chunks = split_into_chunks(extracted, META)
    assert [c.metadata["page_number"] for c in chunks] == [1, 2]
    assert chunks[1].metadata["section"] == "1. DELAIS"  # heading persists across the page break


def test_long_section_is_split_and_every_piece_keeps_the_header():
    body = " ".join(f"Phrase numéro {i} sur les congés." for i in range(80))
    chunks = split_into_chunks(_doc((None, f"## Section longue\n{body}")), META, chunk_size=300, chunk_overlap=30)
    assert len(chunks) > 3
    assert all(c.page_content.startswith("Section longue\n") for c in chunks)
    assert all(len(c.page_content) <= 300 + len("Section longue\n") + 5 for c in chunks)


def test_chunk_ids_are_deterministic():
    extracted = _doc((None, "## A\ntexte un\n## B\ntexte deux"))
    first = [c.metadata["chunk_id"] for c in split_into_chunks(extracted, META)]
    assert first == [c.metadata["chunk_id"] for c in split_into_chunks(extracted, META)]


def test_document_without_headings_still_yields_chunks_titled_by_filename():
    chunks = split_into_chunks(_doc((None, "juste du texte sans titre")), META)
    assert len(chunks) == 1 and chunks[0].metadata["title"] == "politique conges"


# ----- embeddings
def test_hashing_embeddings_are_deterministic_and_normalised():
    emb = HashingEmbeddings()
    v1, v2 = emb.embed_query("jours de congés payés"), emb.embed_query("jours de congés payés")
    assert v1 == v2 and len(v1) == HASHING_DIM
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0, rel_tol=1e-9)


def test_similar_texts_are_closer_than_unrelated_ones():
    emb = HashingEmbeddings()
    cos = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731 - vectors are unit length
    q = emb.embed_query("combien de jours de télétravail par semaine")
    near = emb.embed_query("Le télétravail est limité à 2 jours par semaine.")
    far = emb.embed_query("Le module de formation à la sécurité dure 45 minutes.")
    assert cos(q, near) > cos(q, far)


def test_text_without_content_terms_never_gives_a_zero_vector():
    vector = HashingEmbeddings().embed_query("et le de la ?")
    assert any(vector) and math.isclose(sum(x * x for x in vector), 1.0)

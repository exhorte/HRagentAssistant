import pytest

from hr_assistant.demo import write_docx, write_pdf
from hr_assistant.document_loader import (
    DocumentError, ExtractionError, UnsupportedFormatError, detect_type, extract_text_from_bytes, load_document,
    sanitize_filename, validate_upload,
)


def test_txt_utf8_and_cp1252_are_decoded():
    assert extract_text_from_bytes("a.txt", "Congé payé : 25 jours".encode("utf-8")).text == "Congé payé : 25 jours"
    assert extract_text_from_bytes("a.txt", "Congé payé".encode("cp1252")).text == "Congé payé"


def test_empty_and_binary_txt_are_rejected():
    with pytest.raises(ExtractionError):
        extract_text_from_bytes("a.txt", b"")
    with pytest.raises(ExtractionError):
        extract_text_from_bytes("a.txt", b"abc\x00\x01binary")
    with pytest.raises(ExtractionError):
        extract_text_from_bytes("a.txt", b"   \n  ")


@pytest.mark.parametrize("name", ["data.csv", "image.png", "archive.zip", "noextension", "macro.docm"])
def test_unsupported_formats_are_reported_not_silently_read(name):
    with pytest.raises(UnsupportedFormatError) as exc:
        detect_type(name)
    assert "Formats acceptés" in str(exc.value)


def test_pdf_keeps_page_numbers(tmp_path):
    path = tmp_path / "doc.pdf"
    write_pdf(path, [["PREMIERE PAGE", "Texte un"], ["DEUXIEME PAGE", "Texte deux"]])
    extracted = load_document(path)
    assert extracted.doc_type == "pdf"
    assert [p.number for p in extracted.pages] == [1, 2]
    assert "Texte deux" in extracted.pages[1].text


def test_pdf_without_text_layer_is_an_explicit_error(tmp_path):
    path = tmp_path / "scan.pdf"
    write_pdf(path, [[]])
    with pytest.raises(ExtractionError) as exc:
        load_document(path)
    assert "OCR" in str(exc.value)


def test_corrupted_pdf_is_an_explicit_error():
    with pytest.raises(ExtractionError):
        extract_text_from_bytes("broken.pdf", b"%PDF-1.4\nthis is not really a pdf")


def test_docx_headings_become_markers(tmp_path):
    path = tmp_path / "doc.docx"
    write_docx(path, "Titre", [("1. Congés", ["25 jours par an."]), ("2. Report", ["5 jours."])])
    text = load_document(path).text
    assert "## 1. Congés" in text and "25 jours par an." in text


def test_corrupted_docx_is_an_explicit_error():
    with pytest.raises(ExtractionError):
        extract_text_from_bytes("broken.docx", b"PK\x03\x04not a real zip archive")


def test_signature_must_match_extension():
    with pytest.raises(ExtractionError):
        validate_upload("fake.pdf", b"just text, not a pdf", 1024)
    with pytest.raises(ExtractionError):
        validate_upload("fake.docx", b"just text", 1024)


def test_size_limit():
    with pytest.raises(DocumentError) as exc:
        validate_upload("big.txt", b"x" * (3 * 1024 * 1024), 1024 * 1024)
    assert "volumineux" in str(exc.value)


def test_filename_sanitising_blocks_path_traversal():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("C:\\Users\\x\\secret doc (1).pdf") == "secret doc (1).pdf"
    assert sanitize_filename("..") == "document"
    assert "/" not in sanitize_filename("a/b\\c<>|.txt")

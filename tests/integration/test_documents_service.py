import pytest

from hr_assistant import auth
from hr_assistant.auth import PermissionDenied
from hr_assistant.demo import write_docx, write_pdf
from hr_assistant.document_loader import DocumentError, ExtractionError, UnsupportedFormatError
from hr_assistant.documents import DocumentNotFound, DuplicateDocumentError

pytestmark = pytest.mark.integration

TEXT = "## POLITIQUE DE FORMATION\nChaque salarié a droit à 3 jours de formation par an.".encode("utf-8")


def test_import_indexes_and_records_metadata(app, claire):
    record = app.documents.import_bytes(claire, "formation.txt", TEXT, version="1.0", family="formation")
    assert record.indexing_status == "indexed" and record.chunk_count >= 1
    assert (record.doc_type, record.version, record.status, record.language) == ("txt", "1.0", "active", "fr")
    assert record.imported_by == claire.id and record.imported_at
    assert app.index.count(record.id) == record.chunk_count


def test_pdf_and_docx_can_be_imported(app, claire, tmp_path):
    write_pdf(tmp_path / "a.pdf", [["TITRE", "Deux jours de formation."]])
    write_docx(tmp_path / "b.docx", "Titre", [("1. Formation", ["Trois jours par an."])])
    pdf = app.documents.import_bytes(claire, "a.pdf", (tmp_path / "a.pdf").read_bytes())
    docx = app.documents.import_bytes(claire, "b.docx", (tmp_path / "b.docx").read_bytes())
    assert pdf.indexing_status == docx.indexing_status == "indexed"
    assert pdf.page_count == 1


def test_duplicate_content_is_not_stored_twice(app, claire):
    app.documents.import_bytes(claire, "a.txt", TEXT)
    with pytest.raises(DuplicateDocumentError):
        app.documents.import_bytes(claire, "copie.txt", TEXT)
    assert len(app.documents.list_documents(claire)) == 1


def test_unsupported_format_is_refused_before_anything_is_stored(app, claire, settings):
    with pytest.raises(UnsupportedFormatError):
        app.documents.import_bytes(claire, "data.csv", b"a,b\n1,2")
    assert app.documents.list_documents(claire) == []
    assert not settings.upload_dir.exists() or not any(settings.upload_dir.iterdir())


def test_extraction_failure_is_recorded_and_visible(app, claire):
    with pytest.raises(ExtractionError):
        app.documents.import_bytes(claire, "scan.pdf", b"%PDF-1.4\nnot really a pdf")
    (record,) = app.documents.list_documents(claire)
    assert record.indexing_status == "failed" and record.indexing_error
    assert app.index.count(record.id) == 0


def test_failed_import_can_be_retried_with_the_same_file(app, claire, monkeypatch):
    original = app.index.add_chunks
    monkeypatch.setattr(app.index, "add_chunks", lambda chunks: (_ for _ in ()).throw(RuntimeError("embedding API down")))
    with pytest.raises(DocumentError) as exc:
        app.documents.import_bytes(claire, "a.txt", TEXT)
    assert "embedding API down" not in str(exc.value)  # no internal detail in the user-facing message
    assert app.documents.list_documents(claire)[0].indexing_status == "failed"

    monkeypatch.setattr(app.index, "add_chunks", original)
    record = app.documents.import_bytes(claire, "a.txt", TEXT)
    assert record.indexing_status == "indexed"
    assert len(app.documents.list_documents(claire)) == 1


def test_remove_deletes_registry_file_and_vectors(app, claire, settings):
    record = app.documents.import_bytes(claire, "a.txt", TEXT)
    assert any(settings.upload_dir.iterdir())
    app.documents.remove(claire, record.id)
    assert app.documents.list_documents(claire) == []
    assert app.index.count(record.id) == 0
    assert not any(settings.upload_dir.iterdir())
    with pytest.raises(DocumentNotFound):
        app.documents.remove(claire, record.id)


def test_reindex_rebuilds_vectors_from_the_stored_file(app, claire):
    record = app.documents.import_bytes(claire, "a.txt", TEXT)
    app.index.delete_document(record.id)
    assert app.index.count(record.id) == 0
    again = app.documents.reindex(claire, record.id)
    assert again.indexing_status == "indexed" and app.index.count(record.id) == again.chunk_count


def test_marking_a_document_obsolete_removes_it_from_answers(app, claire, alice):
    record = app.documents.import_bytes(claire, "formation.txt", TEXT, family="formation")
    question = "Combien de jours de formation par an ?"
    assert app.engine.answer(alice, question).status == "answered"
    app.documents.update_metadata(claire, record.id, status="obsolete")
    assert app.engine.answer(alice, question).status == "abstained"


def test_only_hr_manages_documents(app, claire, alice):
    with pytest.raises(PermissionDenied):
        app.documents.import_bytes(alice, "a.txt", TEXT)
    record = app.documents.import_bytes(claire, "a.txt", TEXT)
    for action in (lambda: app.documents.remove(alice, record.id), lambda: app.documents.reindex(alice, record.id)):
        with pytest.raises(PermissionDenied):
            action()


def test_hr_only_documents_are_invisible_to_employees(app, claire, alice):
    secret = app.documents.import_bytes(claire, "grille.txt", "## GRILLE\nNiveau A : 30 000 euros.".encode(), access_level=auth.ACCESS_HR_ONLY)
    assert [d.id for d in app.documents.list_documents(claire)] == [secret.id]
    assert app.documents.list_documents(alice) == []
    with pytest.raises(DocumentNotFound):
        app.documents.get(alice, secret.id)


def test_reindex_all(app, claire):
    app.documents.import_bytes(claire, "a.txt", TEXT)
    assert app.documents.reindex_all(claire) == (1, 0)

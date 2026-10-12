"""Step 2: chop the document into small, searchable, *citable* chunks.

Strategy (simple and configurable):
  1. detect headings (Markdown ``#``/``##``, or ALL-CAPS lines such as
     "1. LEAVE POLICY") and group the text into sections;
  2. keep a section whole if it fits in ``chunk_size`` - HR policy sections are
     short and self-contained, so cutting them mid-rule hurts answers;
  3. otherwise split the section with ``RecursiveCharacterTextSplitter`` and
     repeat the header at the start of every piece so each chunk stays
     understandable on its own.

The *header* of a chunk is ``<document title>\n<section heading>``; it is stored
in the ``header`` metadata so it can be stripped when quoting the passage.

Every chunk carries the document metadata plus ``page_number`` (PDF only),
``section`` and ``chunk_index`` so a citation can point back to its source.
"""

from __future__ import annotations

import re
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hr_assistant.document_loader import ExtractedDocument
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_CHUNK_NAMESPACE = uuid.UUID("6f0f8a1e-2d5b-4f43-9a3a-3c2d8b7e1a10")


def _heading_of(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    match = _MD_HEADING.match(stripped)
    if match:
        return match.group(1).strip()
    letters = [c for c in stripped if c.isalpha()]
    if len(letters) >= 4 and len(stripped) <= 80 and not any(c.islower() for c in letters):
        return stripped
    return None


def _sections(extracted: ExtractedDocument) -> list[tuple[int | None, str | None, str]]:
    """Return ``(page_number, heading, body)`` tuples; the heading carries across pages."""
    out: list[tuple[int | None, str | None, str]] = []
    heading: str | None = None
    for page in extracted.pages:
        body: list[str] = []

        def flush() -> None:
            text = "\n".join(body).strip()
            if text:
                out.append((page.number, heading, text))
            body.clear()

        for line in page.text.splitlines():
            found = _heading_of(line)
            if found:
                flush()
                heading = found
            else:
                body.append(line.rstrip())
        flush()
    return out


def split_into_chunks(
    extracted: ExtractedDocument,
    metadata: dict,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Split an extracted document into LangChain ``Document`` chunks with citation metadata."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    document_id = str(metadata.get("document_id", ""))
    chunks: list[Document] = []
    sections = _sections(extracted)
    # Every chunk starts with the document title (then its section heading) so that a passage such as
    # "25 jours ouvrés par an" stays tied to "POLITIQUE DE CONGÉS" both for retrieval and for display.
    title = next((h for _, h, _ in sections if h), None) or str(metadata.get("document_name", "")).rsplit(".", 1)[0].replace("_", " ")

    for page_number, heading, body in sections:
        header = "\n".join(h for h in dict.fromkeys([title, heading]) if h)
        prefix = f"{header}\n" if header else ""
        text = f"{prefix}{body}"
        pieces = [text] if len(text) <= chunk_size else [prefix + p for p in splitter.split_text(body)]
        for piece in pieces:
            piece = piece.strip()
            if not piece or piece == header:
                continue
            index = len(chunks)
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        **metadata,
                        "page_number": page_number,
                        "section": heading,
                        "title": title,
                        "header": header,
                        "chunk_index": index,
                        "chunk_id": str(uuid.uuid5(_CHUNK_NAMESPACE, f"{document_id}:{index}")),
                    },
                )
            )
    logger.info("Split document %s into %d chunk(s)", document_id[:8], len(chunks))
    return chunks

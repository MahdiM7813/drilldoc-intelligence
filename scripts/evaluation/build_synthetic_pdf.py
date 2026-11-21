"""Build a deterministic, synthetic-only PDF fixture without extra packages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pypdf import PdfReader

PROJECT = Path(__file__).resolve().parents[2]


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_bytes(pages: list[str]) -> bytes:
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    }
    refs = []
    for index, text in enumerate(pages):
        page_id, content_id = 4 + 2 * index, 5 + 2 * index
        refs.append(f"{page_id} 0 R")
        body = text.encode("cp1252")
        objects[page_id] = f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode()
        objects[content_id] = b"<< /Length " + str(len(body)).encode() + b" >>\nstream\n" + body + b"\nendstream"
    objects[2] = f"<< /Type /Pages /Kids [{' '.join(refs)}] /Count {len(pages)} >>".encode()
    output = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = [0]
    for object_id in range(1, max(objects) + 1):
        offsets.append(len(output))
        output += f"{object_id} 0 obj\n".encode() + objects[object_id] + b"\nendobj\n"
    xref = len(output)
    output += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    for offset in offsets[1:]:
        output += f"{offset:010d} 00000 n \n".encode()
    output += f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return output


def build(corpus_path: Path, output_path: Path) -> dict:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8-sig"))
    pages: list[str] = []
    for clause in corpus["clauses"]:
        heading = f"{clause['number']} {clause['title']}"
        lines = [clause["text"][i:i + 80] for i in range(0, len(clause["text"]), 80)]
        stream = f"BT /F1 13 Tf 50 750 Td ({_escape(heading)}) Tj /F1 10 Tf"
        for line in lines:
            stream += f" 0 -24 Td ({_escape(line)}) Tj"
        pages.append(stream + " ET")
    for table in corpus["tables"]:
        title = f"Table {table['number']} - {table['title']}"
        rows = [table["columns"], *table["rows"]]
        widths = [210, 160, 120] if len(rows[0]) == 3 else [260, 220]
        xs = [50]
        for width in widths:
            xs.append(xs[-1] + width)
        stream = f"BT /F1 13 Tf 50 750 Td ({_escape(title)}) Tj ET\n"
        y0, height = 690, 32
        for row_index, row in enumerate(rows):
            y = y0 - row_index * height
            stream += f"{xs[0]} {y} m {xs[-1]} {y} l S\n"
            for x in xs:
                stream += f"{x} {y} m {x} {y - height} l S\n"
            for column, cell in enumerate(row):
                stream += f"BT /F1 10 Tf {xs[column] + 6} {y - 20} Td ({_escape(str(cell))}) Tj ET\n"
        y = y0 - len(rows) * height
        stream += f"{xs[0]} {y} m {xs[-1]} {y} l S\n"
        pages.append(stream)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_pdf_bytes(pages))
    reader = PdfReader(str(output_path))
    if len(reader.pages) != len(pages):
        raise RuntimeError("PDF page-count verification failed")
    return {"output": str(output_path), "pages": len(reader.pages), "bytes": output_path.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=PROJECT / "tests/fixtures/synthetic_engineering_corpus.json")
    parser.add_argument("--output", type=Path, required=True, help="Write to an external temporary path")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output == PROJECT or PROJECT in output.parents:
        parser.error("synthetic PDFs must be written outside the repository")
    if output.exists():
        parser.error(f"refusing to overwrite existing PDF: {output}")
    print(json.dumps(build(args.corpus.resolve(), output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


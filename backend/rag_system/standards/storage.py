from __future__ import annotations

import hashlib
from pathlib import Path


class UploadValidationError(ValueError):
    pass


class StandardFileStorage:
    def __init__(self, root: Path, max_upload_bytes: int):
        self.root = root.resolve()
        self.max_upload_bytes = int(max_upload_bytes)
        self.root.mkdir(parents=True, exist_ok=True)

    def validate(self, filename: str, content: bytes) -> None:
        if Path(filename).suffix.casefold() != ".pdf":
            raise UploadValidationError("only PDF files are accepted")
        if not content.startswith(b"%PDF-"):
            raise UploadValidationError("file content is not a PDF")
        if not content or len(content) > self.max_upload_bytes:
            raise UploadValidationError("PDF size is invalid or exceeds the configured limit")

    def save(self, standard_id: str, filename: str, content: bytes) -> tuple[Path, str]:
        self.validate(filename, content)
        safe_name = Path(filename).name
        target_dir = (self.root / standard_id).resolve()
        if self.root not in target_dir.parents:
            raise UploadValidationError("invalid standard storage path")
        target_dir.mkdir(parents=True, exist_ok=False)
        target = target_dir / safe_name
        target.write_bytes(content)
        return target, hashlib.sha256(content).hexdigest()

# Data and models

The repository includes fictional synthetic benchmark fixtures in `tests/fixtures/`. It does not include original drilling documents, extracted text, document chunks, vector indexes, databases, chat histories, generated answers, evaluation traces, or model weights. The source-workspace corpus and its derived artifacts were not transferred.

Only process documents you are authorized to access, process, and use. Confirm that you also have the rights needed for any intended publication or demonstration. Do not commit customer, well, operational, or otherwise confidential material.

## External storage

- `DRILLDOC_DATA_DIR` stores chat data, standards registry/uploads, generated chunks, indexes, and optional traces. If unset, the application uses the operating system's user data directory.
- `DRILLDOC_MODEL_DIR` identifies local model files. If unset, model files are expected under the data root's `models/` directory.
- `DRILLDOC_VECTOR_INDEX_DIR` selects the Chroma index location; RAG requires a compatible index.

Keep PDFs, extracted text, indexes, databases, model caches, and generated outputs outside the repository. Use a dedicated directory with appropriate access controls and retention for local data.

## Models and providers

No model weights are bundled. Users provide a locally available BGE-M3 embedding model and a Llama-compatible GGUF for API answer generation, and must review the applicable model licenses and terms. The API uses llama.cpp for its reasoning and formatting passes. Model paths are configured locally and must not be committed. See [setup](setup.md) for instructions.

## Trace handling

Forensic traces can contain user questions, selected document context, prompts, raw model responses, citations, and final answers. Treat traces as document-derived sensitive data: keep them outside the repository, restrict access, retain them only as needed, and delete them according to your data-handling requirements. Tracing is optional; do not enable it for confidential material unless the storage and retention implications are understood. The project does not provide enterprise-grade privacy controls.

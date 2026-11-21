# Security

Do not commit credentials, private documents, customer/well identifiers, chat databases, logs, traces, model weights, indexes, or extracted content. Keep runtime data and caches outside the repository.

This is an experimental portfolio project and has not undergone a formal security audit. A local synthetic smoke test and automated tests are not a security review. Do not treat API liveness as RAG readiness; check `/ready` before using the chat API. Dependency resolution and clean-install reproducibility remain unverified.

Report suspected vulnerabilities privately to the project maintainer. Do not include real drilling documents or credentials in issue reports.

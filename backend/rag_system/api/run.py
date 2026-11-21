from __future__ import annotations

import uvicorn

from .app import DEFAULT_API_CONFIG, load_api_config


def main() -> None:
    config = load_api_config(DEFAULT_API_CONFIG)
    uvicorn.run("rag_system.api.app:app", host=str(config["host"]), port=int(config["port"]), reload=False)


if __name__ == "__main__":
    main()

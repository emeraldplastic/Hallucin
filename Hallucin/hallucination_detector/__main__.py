from __future__ import annotations

import os

from .webapp import create_app


def main() -> None:
    app = create_app()
    host = os.getenv("HALLUCIN_HOST", "127.0.0.1")
    port = int(os.getenv("HALLUCIN_PORT", "8000"))
    debug = os.getenv("HALLUCIN_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()

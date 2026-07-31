from __future__ import annotations

import argparse
import json
import os
import sys

from .detector import detect
from .webapp import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Hallucin Studio CLI & Web Application")
    parser.add_argument("--context", type=str, help="Source context text or file path for direct CLI analysis")
    parser.add_argument("--response", type=str, help="Model response text or file path for direct CLI analysis")
    parser.add_argument("--json", action="store_true", help="Output analysis report in JSON format")
    parser.add_argument("--output", type=str, help="Path to save JSON or text report")
    parser.add_argument("--host", type=str, default=os.getenv("HALLUCIN_HOST", "127.0.0.1"), help="Bind host for web server")
    parser.add_argument("--port", type=int, default=int(os.getenv("HALLUCIN_PORT", "8000")), help="Bind port for web server")

    args = parser.parse_args()

    if args.context and args.response:
        ctx = args.context
        if os.path.exists(ctx):
            with open(ctx, "r", encoding="utf-8") as f:
                ctx = f.read()

        resp = args.response
        if os.path.exists(resp):
            with open(resp, "r", encoding="utf-8") as f:
                resp = f.read()

        res = detect(context=ctx, response=resp)

        if args.json:
            out_data = {
                "score": res.score,
                "elapsed_ms": res.elapsed_ms,
                "counts": {
                    "supported": len(res.supported_claims),
                    "partial": len(res.partial_claims),
                    "unsupported": len(res.flagged_claims),
                },
                "claims": [
                    {
                        "claim": c.claim,
                        "label": c.label,
                        "score": c.score,
                        "best_match": c.best_match,
                    }
                    for c in res.claims
                ],
            }
            output_str = json.dumps(out_data, indent=2)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(output_str)
            print(output_str)
        else:
            res.report()
        return

    app = create_app()
    debug = os.getenv("HALLUCIN_DEBUG", "0") == "1"
    app.run(host=args.host, port=args.port, debug=debug)


if __name__ == "__main__":
    main()


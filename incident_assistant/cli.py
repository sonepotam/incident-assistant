"""CLI: incident text (argument, --file or stdin) -> triage JSON on stdout.

Exit codes: 0 ok, 1 bad input/config, 2 model failed to produce valid output.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace

from .agent import IncidentAgent, TriageError
from .config import Settings
from .llm import LLMError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="incident-assistant", description="Triage a production incident.")
    parser.add_argument("text", nargs="?", help="incident description (or use --file / stdin)")
    parser.add_argument("-f", "--file", help="read incident description from a file")
    parser.add_argument("--provider", choices=["auto", "anthropic", "openai", "mock"], help="override LLM_PROVIDER")
    parser.add_argument("--trace", action="store_true", help="print stage trace (attempts, retrieval) to stderr")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else os.getenv("LOG_LEVEL", "WARNING").upper(),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            text = fh.read()
    elif args.text:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.error("provide incident text as an argument, via --file, or on stdin")

    settings = Settings.from_env()
    if args.provider:
        settings = replace(settings, provider=args.provider)

    try:
        agent = IncidentAgent.from_settings(settings)
        outcome = agent.run(text)
    except (ValueError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except TriageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(json.dumps(exc.trace, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1 if isinstance(exc.__cause__, LLMError) else 2

    print(outcome.result.model_dump_json(indent=2))
    if args.trace:
        print(json.dumps(outcome.trace, indent=2, ensure_ascii=False), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

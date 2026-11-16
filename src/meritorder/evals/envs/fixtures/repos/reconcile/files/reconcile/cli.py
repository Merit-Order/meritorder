"""Command line entry point: ``reconcile nightly`` and ``reconcile check``."""

from __future__ import annotations

import argparse
import sys

from . import jobs

PROFILES = ["production", "staging", "dev"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reconcile")
    sub = parser.add_subparsers(dest="command", required=True)
    nightly = sub.add_parser("nightly", help="reconcile yesterday's statement and store the report")
    nightly.add_argument("--profile", default="production", choices=PROFILES)
    check = sub.add_parser("check", help="reconcile one day without storing anything")
    check.add_argument("--day", required=True, help="YYYY-MM-DD")
    check.add_argument("--profile", default="dev", choices=PROFILES)
    args = parser.parse_args(argv)
    report = jobs.nightly(args.profile) if args.command == "nightly" else jobs.check(args.day, args.profile)
    print(report.summary())
    return 0 if report.complete else 1


if __name__ == "__main__":
    sys.exit(main())

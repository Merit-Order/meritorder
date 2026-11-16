"""Matching statement lines to ledger entries."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Report:
    matched: int = 0
    unmatched_bank: list = field(default_factory=list)
    unmatched_ledger: list = field(default_factory=list)
    complete: bool = True

    def summary(self) -> str:
        return f"matched={self.matched} bank_only={len(self.unmatched_bank)} ledger_only={len(self.unmatched_ledger)}"


def match_lines(bank: list[dict], ledger: list[dict]) -> Report:
    """Pair lines with the same reference and amount; everything else is reported."""
    remaining = {(e["reference"], e["amount_cents"]): e for e in ledger}
    report = Report()
    for line in bank:
        if remaining.pop((line["reference"], line["amount_cents"]), None) is not None:
            report.matched += 1
        else:
            report.unmatched_bank.append(line)
    report.unmatched_ledger = list(remaining.values())
    return report

# reconcile

Matches the bank's statement lines against our ledger every night and reports what doesn't match.

    reconcile nightly --profile production
    reconcile check --day 2026-03-02 --profile staging

Settings live in `reconcile/profiles.py` (see `reconcile/config.py` for how profiles combine).
Operations notes: `docs/operations.md`.

Run the tests with `python -m unittest discover -s tests`.

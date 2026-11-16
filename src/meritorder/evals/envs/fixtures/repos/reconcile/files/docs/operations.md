# Operating the nightly reconciliation

_Last reviewed: May 2024._

The scheduler runs `reconcile nightly --profile production` at 02:15 UTC.

## Bank feed outages

The bank feed times out now and then. The job retries a failed fetch 3 times with exponential
backoff (1 s, 2 s, 4 s): a fetch gives up after 4 attempts and about 7 seconds of waiting. When it
gives up, the run is marked `partial` and the on-call engineer gets a ticket.

## Ledger write failures

Writes to the ledger are retried by the database driver. If the report can't be stored, the job
exits non-zero and the scheduler retries the whole run once after 30 minutes.

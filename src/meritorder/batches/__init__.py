"""Provider batch APIs: work that can wait up to a day, for about half the price (docs/02-request-flow.md#batches-work-that-can-wait).

Clients use OpenAI's Batch API (``/v1/files``, ``/v1/batches``) or Anthropic's Message Batches (``/v1/messages/batches``)
as they would at the provider. The gateway routes each request of a batch on its own, at batch prices, splits the batch
into one job per provider (and model, where the provider takes one model a job), and joins the results back in the
client's format when it asks for them. The worker polls the jobs until they end and writes one ledger row per request.

* ``store``: the tables (metadata only, plus encrypted uploads until they're submitted), as mixins of the stores.
* ``providers``: each provider's batch API.
* ``service``: creating, reading, cancelling and joining batches; settling their results into the ledger.
* ``poller``: the worker's loop over provider jobs.
* ``api``: the HTTP endpoints.
"""

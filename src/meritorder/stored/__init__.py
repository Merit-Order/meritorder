"""The Responses API's stored mode (decision 51, docs/04-caching-and-sessions.md#stateful-apis): opt-in per team, where
the gateway holds conversation content.

With a team's ``stored_responses`` on, ``/v1/responses`` keeps each response (``store: true``, OpenAI's default), so
clients can chain turns with ``previous_response_id`` or a ``conversation`` and read responses back. The gateway holds
the conversation, not the provider: every turn is rebuilt from what it stored and routed like a stateless request, so
``auto`` can send turn 2 to another model or provider than turn 1. Content is sealed (AES-GCM) with a key from the
environment or a file, scoped to the person and team that made it, and deleted after the team's retention.

* ``store``: the tables (content sealed; routing metadata and sizes in the clear), as mixins of the stores.
* ``service``: who may read what, rebuilding a turn's input, sealing and saving, background responses, purging.
* ``api``: ``GET``/``DELETE /v1/responses/{id}``, ``/input_items``, ``/cancel`` and ``/v1/conversations``.
"""

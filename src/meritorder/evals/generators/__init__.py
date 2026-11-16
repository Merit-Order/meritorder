"""Deterministic prompt generators for eval tasks.

A task.yaml can say ``generate: meritorder.evals.generators.<module>:<callable>`` with ``generate_args``; the callable
returns ``{"prompt": str, "checks": [...]}``. Generators use only the standard library and a seeded
``random.Random``, so the same arguments always produce the same prompt and answer.
"""

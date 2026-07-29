"""Capability benchmark: tool-calling and structured-output for proxy LLMs.

This suite measures the two LLM capabilities BioGuider actually depends on:

* **tool calling** — the PEO agents (``*_plan_step`` / ``*_execute_step``) rely on
  the backing model emitting well-formed tool calls with correct arguments.
* **structured output** — every ``Evaluation*Task`` parses the model's reply into a
  typed dict; a model that can't honor a JSON schema breaks the whole pipeline.

Scoring is deterministic (no LLM judge) so runs are cheap and reproducible.
Entry point: ``benchmark/test_capabilities.py``.
"""

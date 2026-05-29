"""
Observability utilities: Logfire setup and error introspection helpers.
"""

import logfire
from openai import OpenAI


def configure_logfire(openai_client: OpenAI | None = None) -> None:
    logfire.configure(
        service_name="diy-pipeline",
        service_version="0.1.0",
        send_to_logfire="if-token-present",
        console=False,
    )
    if openai_client:
        logfire.instrument_openai(openai_client)

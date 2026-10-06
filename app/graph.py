"""LangGraph pipeline definition stub for the report-drafting state machine."""

from typing import Any


def build_pipeline_graph() -> Any:
    """Construct the LangGraph state machine orchestrating vision -> rules -> RAG -> LLM -> verify.

    Returns:
        Compiled LangGraph workflow runnable (stub).
    """
    # Stub implementation: StateGraph nodes and conditional edges (retry / fallback)
    raise NotImplementedError("build_pipeline_graph stub - to be implemented in pipeline phase")

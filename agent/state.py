"""Typed state shared by all research graph nodes."""

from __future__ import annotations

from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class SearchResult(TypedDict):
    """One MCP search or page-fetch result."""

    query: str
    tool: str
    content: str
    urls: list[str]
    error: str


class ResearchState(TypedDict, total=False):
    """State persisted by LangGraph for one research conversation."""

    question: str
    plan: str
    sub_questions: list[str]
    search_queries: list[str]
    search_results: list[SearchResult]
    analyzed_result_count: int
    notes: str
    draft_report: str
    final_report: str
    iteration_count: int
    messages: Annotated[list[AnyMessage], add_messages]
    needs_more: NotRequired[bool]
    reflection: NotRequired[str]

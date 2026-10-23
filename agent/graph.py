"""LangGraph construction and bounded search-reflection workflow."""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import analyzer, planner, reflector, searcher, writer
from agent.state import ResearchState
from config import (
    DEFAULT_MAX_SEARCH_ITERATIONS,
    MAX_ALLOWED_SEARCH_ITERATIONS,
    load_mcp_config,
)
from mcp_client import find_search_tools

logger = logging.getLogger(__name__)
_MEMORY_SAVER = MemorySaver()


def build_research_graph(
    llm: Any,
    tools: list[BaseTool],
    *,
    max_search_iterations: int = DEFAULT_MAX_SEARCH_ITERATIONS,
    checkpointer: MemorySaver | None = None,
    mcp_config: dict[str, Any] | None = None,
) -> Any:
    """Build and compile the research StateGraph with thread-scoped memory."""
    if not 1 <= max_search_iterations <= MAX_ALLOWED_SEARCH_ITERATIONS:
        raise ValueError(
            f"max_search_iterations must be between 1 and "
            f"{MAX_ALLOWED_SEARCH_ITERATIONS}."
        )
    if not tools:
        raise ValueError("At least one MCP tool is required to build the graph.")

    config = mcp_config or load_mcp_config()
    search_tool_names = config.get("search_tool_names", [])
    if not find_search_tools(tools, search_tool_names):
        available = ", ".join(tool.name for tool in tools)
        raise ValueError(
            "No MCP search tool was found. Configure search_tool_names in "
            f"mcp_servers.json. Available tools: {available}"
        )

    try:
        bound_llm = llm.bind_tools(tools)
    except Exception as exc:
        logger.exception("Could not bind MCP tools to the Groq model")
        raise RuntimeError(
            "The Groq model could not bind the configured MCP tools. "
            "Check the server tool schemas and selected model."
        ) from exc

    workflow = StateGraph(ResearchState)
    workflow.add_node("planner", partial(planner, llm=llm))
    workflow.add_node(
        "searcher",
        partial(
            searcher,
            bound_llm=bound_llm,
            tools=tools,
            search_tool_names=search_tool_names,
            fetch_tool_names=config.get("fetch_tool_names", []),
        ),
    )
    workflow.add_node("analyzer", partial(analyzer, llm=llm))
    workflow.add_node(
        "reflector",
        partial(
            reflector,
            llm=llm,
            max_search_iterations=max_search_iterations,
        ),
    )
    workflow.add_node("writer", partial(writer, llm=llm))

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "searcher")
    workflow.add_edge("searcher", "analyzer")
    workflow.add_edge("analyzer", "reflector")
    workflow.add_conditional_edges(
        "reflector",
        lambda state: "searcher" if state.get("needs_more", False) else "writer",
        {"searcher": "searcher", "writer": "writer"},
    )
    workflow.add_edge("writer", END)

    graph = workflow.compile(checkpointer=checkpointer or _MEMORY_SAVER)
    logger.info(
        "Compiled research graph with maximum %d search iteration(s)",
        max_search_iterations,
    )
    return graph

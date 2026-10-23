"""Async LangGraph node implementations for planning and report generation."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from agent.llm import invoke_with_retry
from agent.prompts import (
    ANALYZER_SYSTEM_PROMPT,
    ANALYZER_USER_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    REFLECTOR_USER_PROMPT,
    SEARCH_TOOL_SYSTEM_PROMPT,
    SEARCH_TOOL_USER_PROMPT,
    WRITER_SYSTEM_PROMPT,
    WRITER_USER_PROMPT,
)
from agent.state import ResearchState, SearchResult
from mcp_client import find_fetch_tools, find_search_tools

logger = logging.getLogger(__name__)
URL_PATTERN = re.compile(r"https?://[^\s<>\]\[\"')]+", re.IGNORECASE)
REFERENCE_PATTERN = re.compile(r"ref://[A-Za-z0-9_-]+", re.IGNORECASE)
MAX_RESULT_CHARS = 2400
MAX_SEARCH_RESULTS = 3
MAX_ANALYSIS_CONTEXT_CHARS = 4000
MAX_REGISTERED_SOURCES = 15


class PlanOutput(BaseModel):
    """Structured research plan from the planner node."""

    plan: str = Field(description="A brief description of the research approach.")
    sub_questions: list[str] = Field(min_length=3, max_length=5)
    search_queries: list[str] = Field(min_length=3, max_length=5)


class AnalysisOutput(BaseModel):
    """Evidence-grounded analyst notes."""

    notes: str = Field(description="Concise findings with inline source numbers.")


class ReflectionOutput(BaseModel):
    """Evidence sufficiency and optional follow-up searches."""

    sufficient: bool
    rationale: str
    follow_up_queries: list[str] = Field(default_factory=list, max_length=3)


def _message_text(message: Any) -> str:
    """Convert a model or MCP content object into displayable text."""
    if isinstance(message, str):
        return message
    if isinstance(message, (dict, list)):
        return json.dumps(message, ensure_ascii=False, default=str)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if text:
                parts.append(str(text))
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            else:
                parts.append(json.dumps(block, ensure_ascii=False, default=str))
        return "\n".join(parts)
    return str(message)


def _urls_in(value: Any) -> list[str]:
    """Extract unique HTTP(S) URLs from tool output, including structured links."""
    serialized = _message_text(value)
    found = [match.rstrip(".,;:!?") for match in URL_PATTERN.findall(serialized)]
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.casefold() in {"url", "link", "uri", "source_url"} and isinstance(
                nested, str
            ):
                found.append(nested.strip())
            elif isinstance(nested, (dict, list, tuple)):
                found.extend(_urls_in(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            if isinstance(nested, (dict, list, tuple)):
                found.extend(_urls_in(nested))

    unique_urls: list[str] = []
    for url in found:
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"} and parsed.netloc and url not in unique_urls:
            unique_urls.append(url)
    return unique_urls


def _tool_name(tool_call: dict[str, Any]) -> str:
    """Read a tool name from a LangChain tool-call dictionary."""
    return str(tool_call.get("name", ""))


def _query_args(tool: BaseTool, query: str) -> dict[str, Any]:
    """Map a query to a common search argument supported by a tool schema."""
    fields = getattr(tool, "args", {})
    if not isinstance(fields, dict):
        fields = {}
    preferred_fields = ("query", "search_query", "q", "text", "keywords")
    for field in preferred_fields:
        if field in fields:
            arguments = {field: query}
            for limit_field in ("max_results", "num_results", "count"):
                if limit_field in fields:
                    arguments[limit_field] = MAX_SEARCH_RESULTS
                    break
            return arguments
    if fields:
        return {next(iter(fields)): query}
    return {"query": query}


def _bounded_search_args(tool: BaseTool, arguments: dict[str, Any]) -> dict[str, Any]:
    """Cap result counts even when the model supplies its own tool arguments."""
    bounded = dict(arguments)
    fields = getattr(tool, "args", {})
    if not isinstance(fields, dict):
        return bounded

    for field in ("max_results", "num_results", "count"):
        if field not in fields:
            continue
        value = bounded.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            bounded[field] = MAX_SEARCH_RESULTS
        else:
            bounded[field] = min(max(value, 1), MAX_SEARCH_RESULTS)
        break
    return bounded


def _query_for_tool_call(
    tool_call: dict[str, Any], pending_queries: list[str], completed_queries: set[str]
) -> str | None:
    """Associate a model-issued search call with one requested query."""
    arguments = tool_call.get("args", {})
    if not isinstance(arguments, dict):
        return None
    requested_query = next(
        (
            str(arguments[field]).strip()
            for field in ("query", "search_query", "q", "text", "keywords")
            if arguments.get(field)
        ),
        "",
    )
    if requested_query:
        normalized_request = requested_query.casefold()
        for query in pending_queries:
            if query.casefold() == normalized_request and query not in completed_queries:
                return query
        return None
    return next(
        (query for query in pending_queries if query not in completed_queries),
        None,
    )


def _url_args(tool: BaseTool, url: str) -> dict[str, Any]:
    """Map a source URL to a common page-reader argument."""
    fields = getattr(tool, "args", {})
    if not isinstance(fields, dict):
        fields = {}
    for field in ("url", "uri", "page_url", "link"):
        if field in fields:
            return {field: url}
    if fields:
        return {next(iter(fields)): url}
    return {"url": url}


def _result_record(
    query: str, tool_name: str, output: Any, error: str = ""
) -> SearchResult:
    """Build a consistently shaped result from MCP output."""
    return {
        "query": query,
        "tool": tool_name,
        "content": _message_text(output)[:MAX_RESULT_CHARS],
        "urls": _urls_in(output),
        "error": error,
    }


def _format_results_for_prompt(
    results: list[SearchResult],
    *,
    citation_results: list[SearchResult] | None = None,
) -> str:
    """Format bounded evidence using stable citation numbers across all results."""
    registry: dict[str, int] = {}
    for result in citation_results or results:
        for url in result["urls"]:
            if url not in registry and len(registry) < MAX_REGISTERED_SOURCES:
                registry[url] = len(registry) + 1

    blocks: list[str] = []
    remaining_chars = MAX_ANALYSIS_CONTEXT_CHARS
    for result in results:
        citation_numbers = [
            f"[{registry[url]}] {url}"
            for url in result["urls"]
            if url in registry
        ]
        content = result["content"][:MAX_RESULT_CHARS]
        record = {
            "query": result["query"],
            "tool": result["tool"],
            "sources": citation_numbers,
            "content": content,
            "error": result["error"],
        }
        block = json.dumps(record, ensure_ascii=False)
        while len(block) > remaining_chars and content:
            content = content[: max(0, len(content) - (len(block) - remaining_chars))]
            record["content"] = content
            block = json.dumps(record, ensure_ascii=False)
        if len(block) > remaining_chars:
            break
        blocks.append(block)
        remaining_chars -= len(block) + 2
        if remaining_chars <= 0:
            break
    return "\n\n".join(blocks) or "No search results were returned."


async def planner(state: ResearchState, *, llm: Any) -> dict[str, Any]:
    """Create a focused plan and initial set of web searches."""
    history = "\n".join(
        f"{getattr(message, 'type', 'message')}: "
        f"{_message_text(getattr(message, 'content', message))[:500]}"
        for message in state.get("messages", [])[-8:]
    )
    prompt = PLANNER_USER_PROMPT.format(
        question=state["question"], history=history or "No earlier messages."
    )
    structured_llm = llm.with_structured_output(PlanOutput)
    output = await invoke_with_retry(
        structured_llm.ainvoke,
        [SystemMessage(content=PLANNER_SYSTEM_PROMPT), HumanMessage(content=prompt)],
        operation="planner",
    )
    logger.info("Created plan with %d search queries", len(output.search_queries))
    return {
        "plan": output.plan,
        "sub_questions": output.sub_questions,
        "search_queries": output.search_queries,
        "search_results": [],
        "analyzed_result_count": 0,
        "notes": "",
        "draft_report": "",
        "final_report": "",
        "iteration_count": 0,
        "needs_more": False,
        "reflection": "",
    }


async def searcher(
    state: ResearchState,
    *,
    bound_llm: Any,
    tools: list[BaseTool],
    search_tool_names: list[str],
    fetch_tool_names: list[str],
) -> dict[str, Any]:
    """Select MCP search calls in one model request, then execute each via MCP."""
    search_tools = find_search_tools(tools, search_tool_names)
    if not search_tools:
        raise RuntimeError(
            "No MCP web-search tool was found. Update search_tool_names or the "
            "server configuration in mcp_servers.json."
        )
    fetch_tools = find_fetch_tools(tools, fetch_tool_names)
    tool_by_name = {tool.name: tool for tool in tools}
    current_queries = state.get("search_queries", [])
    existing_results = list(state.get("search_results", []))
    previous_queries = {result["query"] for result in existing_results}
    pending_queries = [query for query in current_queries if query not in previous_queries]
    new_results: list[SearchResult] = []
    completed_queries: set[str] = set()
    decision: Any = None
    if pending_queries:
        try:
            decision = await invoke_with_retry(
                bound_llm.ainvoke,
                [
                    SystemMessage(content=SEARCH_TOOL_SYSTEM_PROMPT),
                    HumanMessage(
                        content=SEARCH_TOOL_USER_PROMPT.format(
                            queries="\n".join(
                                f"{index}. {query}"
                                for index, query in enumerate(
                                    pending_queries, start=1
                                )
                            )
                        )
                    ),
                ],
                operation="search tool selection",
            )
        except Exception as exc:
            logger.warning("LLM search-tool selection failed: %s", exc)

    for tool_call in getattr(decision, "tool_calls", []) if decision else []:
        called_tool_name = _tool_name(tool_call)
        tool = tool_by_name.get(called_tool_name)
        if tool is None:
            logger.warning("Model selected unknown MCP tool %r", called_tool_name)
            continue
        if not any(candidate.name == tool.name for candidate in search_tools):
            logger.warning(
                "Ignoring non-search MCP tool call %r from the model",
                called_tool_name,
            )
            continue
        query = _query_for_tool_call(tool_call, pending_queries, completed_queries)
        if query is None:
            logger.warning("Ignoring extra MCP search call without a pending query")
            continue
        try:
            arguments = tool_call.get("args", {})
            if not isinstance(arguments, dict):
                arguments = {}
            output = await tool.ainvoke(_bounded_search_args(tool, arguments))
            new_results.append(_result_record(query, tool.name, output))
            completed_queries.add(query)
        except Exception as exc:
            logger.exception("MCP tool %s failed for query %r", tool.name, query)
            new_results.append(_result_record(query, tool.name, "", error=str(exc)))
            completed_queries.add(query)

    for query in pending_queries:
        if query in completed_queries:
            continue
        search_tool = search_tools[0]
        try:
            output = await search_tool.ainvoke(_query_args(search_tool, query))
            new_results.append(_result_record(query, search_tool.name, output))
            completed_queries.add(query)
        except Exception as exc:
            logger.exception(
                "MCP search tool %s failed for query %r",
                search_tool.name,
                query,
            )
            new_results.append(
                _result_record(query, search_tool.name, "", error=str(exc))
            )
            completed_queries.add(query)

    if fetch_tools:
        page_targets = list(
            dict.fromkeys(
                target
                for result in new_results
                for target in (
                    result["urls"] + REFERENCE_PATTERN.findall(result["content"])
                )
            )
        )
        if page_targets:
            target = page_targets[0]
            fetch_tool = fetch_tools[0]
            query = next(
                (
                    result["query"]
                    for result in new_results
                    if target in result["urls"] or target in result["content"]
                ),
                pending_queries[0],
            )
            try:
                output = await fetch_tool.ainvoke(_url_args(fetch_tool, target))
                new_results.append(_result_record(query, fetch_tool.name, output))
            except Exception as exc:
                logger.warning(
                    "Optional page fetch via %s failed for %s: %s",
                    fetch_tool.name,
                    target,
                    exc,
                )
                new_results.append(
                    _result_record(query, fetch_tool.name, "", error=str(exc))
                )

    logger.info(
        "Search iteration %d completed with %d new result(s)",
        state.get("iteration_count", 0) + 1,
        len(new_results),
    )
    return {
        "search_results": existing_results + new_results,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


async def analyzer(state: ResearchState, *, llm: Any) -> dict[str, Any]:
    """Extract supported findings while preserving source numbers and URLs."""
    all_results = state.get("search_results", [])
    analyzed_result_count = state.get("analyzed_result_count", 0)
    results = all_results[analyzed_result_count:]
    if not any(result["content"].strip() for result in results):
        notes = (
            "No usable web search results were retrieved for the latest search "
            "pass. The report must clearly state any evidence limitations."
        )
        prior_notes = state.get("notes", "").strip()
        combined_notes = f"{prior_notes}\n\n{notes}".strip() if prior_notes else notes
        return {
            "notes": combined_notes,
            "analyzed_result_count": len(all_results),
        }

    prompt = ANALYZER_USER_PROMPT.format(
        question=state["question"],
        results=_format_results_for_prompt(
            results, citation_results=all_results
        ),
    )
    structured_llm = llm.with_structured_output(AnalysisOutput)
    output = await invoke_with_retry(
        structured_llm.ainvoke,
        [SystemMessage(content=ANALYZER_SYSTEM_PROMPT), HumanMessage(content=prompt)],
        operation="analyzer",
    )
    prior_notes = state.get("notes", "").strip()
    notes = f"{prior_notes}\n\n{output.notes}".strip() if prior_notes else output.notes
    return {"notes": notes, "analyzed_result_count": len(all_results)}


async def reflector(
    state: ResearchState, *, llm: Any, max_search_iterations: int
) -> dict[str, Any]:
    """Assess evidence quality and propose additional searches if useful."""
    iteration_count = state.get("iteration_count", 0)
    if iteration_count >= max_search_iterations:
        return {
            "needs_more": False,
            "reflection": "Maximum search iterations reached.",
        }

    prompt = REFLECTOR_USER_PROMPT.format(
        question=state["question"],
        plan=state.get("plan", ""),
        sub_questions="\n".join(
            f"- {sub_question}" for sub_question in state.get("sub_questions", [])
        )
        or "No sub-questions were supplied.",
        notes=state.get("notes", "") or "No supported findings yet.",
        iteration_count=iteration_count,
        max_iterations=max_search_iterations,
    )
    structured_llm = llm.with_structured_output(ReflectionOutput)
    output = await invoke_with_retry(
        structured_llm.ainvoke,
        [
            SystemMessage(content=REFLECTOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ],
        operation="reflector",
    )
    used_queries = {
        result["query"] for result in state.get("search_results", [])
    }
    follow_up_queries = [
        query
        for query in output.follow_up_queries
        if query not in used_queries
    ][:3]
    needs_more = (
        not output.sufficient
        and bool(follow_up_queries)
        and iteration_count < max_search_iterations
    )
    logger.info(
        "Reflection sufficient=%s; follow-up queries=%d",
        output.sufficient,
        len(follow_up_queries) if needs_more else 0,
    )
    return {
        "needs_more": needs_more,
        "reflection": output.rationale,
        "search_queries": follow_up_queries if needs_more else [],
    }


def _source_registry(results: list[SearchResult]) -> list[dict[str, str]]:
    """Build a stable, unique registry from URLs present in MCP results."""
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for result in results:
        for url in result["urls"]:
            if url in seen:
                continue
            seen.add(url)
            if len(sources) >= MAX_REGISTERED_SOURCES:
                return sources
            hostname = urlparse(url).netloc
            sources.append(
                {
                    "number": str(len(sources) + 1),
                    "title": hostname or url,
                    "url": url,
                }
            )
    return sources


def _remove_unknown_citations(report: str, source_count: int) -> str:
    """Replace citation markers outside the retrieved source registry."""
    def validate(match: re.Match[str]) -> str:
        number = int(match.group(1))
        return match.group(0) if 1 <= number <= source_count else "[unverified citation]"

    return re.sub(r"\[(\d+)\]", validate, report)


async def writer(state: ResearchState, *, llm: Any) -> dict[str, str]:
    """Write the report and append a source list created only from retrieved URLs."""
    sources = _source_registry(state.get("search_results", []))
    prompt = WRITER_USER_PROMPT.format(
        question=state["question"],
        notes=state.get("notes", "") or "No verified web findings are available.",
        sources=json.dumps(sources, ensure_ascii=False, indent=2)
        if sources
        else "No source URLs were retrieved.",
    )
    response = await invoke_with_retry(
        llm.ainvoke,
        [SystemMessage(content=WRITER_SYSTEM_PROMPT), HumanMessage(content=prompt)],
        operation="writer",
    )
    draft = _remove_unknown_citations(_message_text(response).strip(), len(sources))
    if not draft:
        raise RuntimeError("The report writer returned an empty response.")

    source_section = "\n\n## Sources\n"
    if sources:
        source_section += "\n".join(
            f"{source['number']}. [{source['title']}]({source['url']})"
            for source in sources
        )
    else:
        source_section += "\nNo source URLs were returned by the search tools."

    final_report = draft + source_section
    logger.info("Generated final research report with %d source(s)", len(sources))
    return {"draft_report": draft, "final_report": final_report}

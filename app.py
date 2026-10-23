"""Streamlit chat interface for the MCP-backed research agent."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

import streamlit as st
from langchain_core.messages import HumanMessage

from agent.graph import build_research_graph
from agent.llm import MissingAPIKeyError, create_llm
from config import (
    DEFAULT_MAX_SEARCH_ITERATIONS,
    DEFAULT_MODEL,
    MAX_ALLOWED_SEARCH_ITERATIONS,
    load_mcp_config,
)
from mcp_client import (
    MCPConnectionError,
    create_mcp_client,
    load_mcp_tools,
)

logger = logging.getLogger(__name__)
MODEL_OPTIONS = [
    DEFAULT_MODEL,
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]
STEP_LABELS = {
    "planner": "Planning research questions and searches",
    "searcher": "Searching the web with MCP",
    "analyzer": "Analyzing retrieved evidence",
    "reflector": "Checking evidence coverage",
    "writer": "Writing the cited report",
}

st.set_page_config(page_title="AI Research Agent", page_icon=":material/travel_explore:")


def _new_session() -> str:
    """Create a new in-memory chat session with a unique graph thread ID."""
    thread_id = str(uuid.uuid4())
    st.session_state.research_sessions[thread_id] = {
        "title": "New research",
        "messages": [],
    }
    return thread_id


def _initialize_state() -> None:
    """Initialize Streamlit session data once per browser session."""
    st.session_state.setdefault("research_sessions", {})
    st.session_state.setdefault("active_thread_id", "")
    st.session_state.setdefault("session_selector", "")
    if not st.session_state.active_thread_id:
        thread_id = _new_session()
        st.session_state.active_thread_id = thread_id
        st.session_state.session_selector = thread_id


def _session_label(thread_id: str) -> str:
    """Return a readable selector label for a research conversation."""
    session = st.session_state.research_sessions[thread_id]
    title = session["title"].strip()
    return f"{title[:55]}{'…' if len(title) > 55 else ''}"


async def _run_research(
    *,
    question: str,
    thread_id: str,
    model_name: str,
    temperature: float,
    max_search_iterations: int,
    api_key: str,
    on_update: Any,
) -> str:
    """Connect to MCP, stream graph updates to the UI, and return the report."""
    llm = create_llm(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
    )
    mcp_config = load_mcp_config()
    client = create_mcp_client(mcp_config)
    tools = await load_mcp_tools(client)
    graph = build_research_graph(
        llm,
        tools,
        max_search_iterations=max_search_iterations,
        mcp_config=mcp_config,
    )
    graph_config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["research-agent"],
        "metadata": {"thread_id": thread_id, "model": model_name},
    }
    async for update in graph.astream(
        {
            "question": question,
            "messages": [HumanMessage(content=question)],
        },
        config=graph_config,
        stream_mode="updates",
    ):
        for node_name, node_update in update.items():
            on_update(node_name, node_update)
            if node_name == "writer" and node_update.get("final_report"):
                return node_update["final_report"]

    raise RuntimeError("The research graph ended without producing a final report.")


def _new_research_button() -> None:
    """Start a fresh conversation while retaining prior session history."""
    thread_id = _new_session()
    st.session_state.active_thread_id = thread_id
    st.session_state.session_selector = thread_id


_initialize_state()

with st.sidebar:
    st.header("Research settings")
    st.selectbox(
        "Groq model",
        options=MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(DEFAULT_MODEL),
        key="model_name",
        help="Choose the Groq-hosted model used for planning and report writing.",
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        key="temperature",
    )
    max_search_iterations = st.slider(
        "Maximum search iterations",
        min_value=1,
        max_value=MAX_ALLOWED_SEARCH_ITERATIONS,
        value=DEFAULT_MAX_SEARCH_ITERATIONS,
        step=1,
        key="max_search_iterations",
        help="Maximum number of search-and-analysis passes (1-3).",
    )
    api_key = st.text_input(
        "Groq API key",
        type="password",
        value="",
        key="groq_api_key",
        help="Optional. If blank, GROQ_API_KEY from .env is used.",
    )

    st.divider()
    st.subheader("Session history")
    st.button(
        "New research",
        icon=":material/add:",
        width="stretch",
        on_click=_new_research_button,
    )

    session_ids = list(st.session_state.research_sessions)
    labels = {thread_id: _session_label(thread_id) for thread_id in session_ids}
    if st.session_state.session_selector not in labels:
        st.session_state.session_selector = st.session_state.active_thread_id
    selected_thread = st.selectbox(
        "Conversations",
        options=session_ids,
        format_func=lambda thread_id: labels[thread_id],
        key="session_selector",
        label_visibility="collapsed",
    )
    st.session_state.active_thread_id = selected_thread

    langsmith_key_available = bool(os.getenv("LANGSMITH_API_KEY"))
    if langsmith_key_available:
        st.caption(
            f"LangSmith tracing: {os.getenv('LANGSMITH_PROJECT', 'research-agent')}"
        )
    else:
        st.caption("LangSmith tracing is disabled until LANGSMITH_API_KEY is set.")

st.title("AI Research Agent")
st.caption(
    "Ask a research question. The agent searches the web, evaluates its evidence, "
    "and returns a cited report."
)

active_session = st.session_state.research_sessions[
    st.session_state.active_thread_id
]
for message in active_session["messages"]:
    with st.chat_message(message["role"]):
        if message.get("report"):
            st.markdown(message["content"])
            st.download_button(
                "Download report",
                data=message["content"],
                file_name="research-report.md",
                mime="text/markdown",
                key=f"download-{message['id']}",
            )
        else:
            st.markdown(message["content"])

question = st.chat_input(
    "What would you like to research?",
    key="research_question",
    max_chars=2000,
    submit_mode="disable",
)

if question:
    clean_question = question.strip()
    if not clean_question:
        st.warning("Enter a research question before submitting.")
        st.stop()

    active_session = st.session_state.research_sessions[
        st.session_state.active_thread_id
    ]
    active_session["title"] = clean_question
    user_message = {"id": str(uuid.uuid4()), "role": "user", "content": clean_question}
    active_session["messages"].append(user_message)
    with st.chat_message("user"):
        st.markdown(clean_question)

    report_message: dict[str, Any] | None = None
    with st.chat_message("assistant"):
        try:
            with st.status("Researching your question", expanded=True) as status:
                def report_progress(node_name: str, node_update: dict[str, Any]) -> None:
                    label = STEP_LABELS.get(node_name, f"Completed {node_name}")
                    details = ""
                    if node_name == "planner":
                        details = node_update.get("plan", "")
                    elif node_name == "searcher":
                        details = (
                            f"Search pass {node_update.get('iteration_count', 0)} "
                            f"completed; {len(node_update.get('search_results', []))} "
                            "result record(s) collected."
                        )
                    elif node_name == "reflector":
                        details = node_update.get("reflection", "")
                    st.write(f"**{label}**")
                    if details:
                        st.caption(details)

                report = asyncio.run(
                    _run_research(
                        question=clean_question,
                        thread_id=st.session_state.active_thread_id,
                        model_name=st.session_state.model_name,
                        temperature=temperature,
                        max_search_iterations=max_search_iterations,
                        api_key=api_key,
                        on_update=report_progress,
                    )
                )
                status.update(label="Research complete", state="complete")
            st.markdown(report)
            report_message = {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": report,
                "report": True,
            }
            st.download_button(
                "Download report",
                data=report,
                file_name="research-report.md",
                mime="text/markdown",
                key=f"download-{report_message['id']}",
            )
        except MissingAPIKeyError as exc:
            logger.warning("Research request missing Groq API key")
            st.error(str(exc), icon=":material/key_off:")
        except MCPConnectionError as exc:
            logger.exception("MCP connection failed")
            st.error(str(exc), icon=":material/wifi_off:")
        except Exception as exc:
            logger.exception("Research request failed")
            st.error(
                f"Research failed: {exc}. Check the app logs, API credentials, "
                "and MCP server configuration, then retry.",
                icon=":material/error:",
            )

    if report_message is not None:
        active_session["messages"].append(report_message)

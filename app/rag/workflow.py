import logging
import random
import re
import time
from threading import Lock
from typing import Literal

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END

from app.core.config import get_settings
from app.rag.state import AgentState
from app.rag.vectorstore import get_retriever


logger = logging.getLogger(__name__)

settings = get_settings()


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
# Your Groq organization currently has a very small OTPM limit.
# Keep completion limits conservative.
FAST_MAX_TOKENS = 80
ANSWER_MAX_TOKENS = 300

# Prevent huge prompts from unnecessarily consuming tokens.
MAX_KB_CONTEXT_CHARS = 12000
MAX_WEB_CONTEXT_CHARS = 12000
MAX_KB_DOCS = 5

# Retry configuration for transient Groq errors.
MAX_LLM_RETRIES = 3
INITIAL_RETRY_DELAY = 2.0
MAX_RETRY_DELAY = 15.0

# Only one Groq request at a time in this process.
# This is especially important because your OTPM limit is low.
_llm_execution_lock = Lock()


# ============================================================
# LAZY CLIENTS
# ============================================================

_llm = None
_fast_llm = None
_web_search = None


def _create_llm(max_tokens: int):
    """
    Create a ChatGroq client.

    IMPORTANT:
    We intentionally use max_tokens instead of max_completion_tokens
    because the installed langchain-groq version treats the latter
    as an unknown parameter and moves it into model_kwargs.
    """

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is missing")

    return ChatGroq(
        model=settings.openai_model,
        temperature=0,
        max_tokens=max_tokens,
        api_key=settings.groq_api_key,
    )


def llm():
    """
    Main answer-generation model.
    """

    global _llm

    if _llm is None:
        _llm = _create_llm(ANSWER_MAX_TOKENS)

    return _llm


def fast_llm():
    """
    Small-output model/client used for classification/grading.
    """

    global _fast_llm

    if _fast_llm is None:
        _fast_llm = _create_llm(FAST_MAX_TOKENS)

    return _fast_llm


def web_search_tool():
    """
    Tavily web search client.
    """

    global _web_search

    if _web_search is None:

        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")

        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )

    return _web_search


# ============================================================
# SAFE GROQ INVOCATION
# ============================================================

def _is_rate_limit_error(exc: Exception) -> bool:
    """
    Detect Groq rate-limit errors without depending on a specific
    exception class because different LangChain/Groq versions can
    wrap provider exceptions differently.
    """

    text = str(exc).lower()

    rate_limit_markers = [
        "429",
        "rate_limit",
        "rate limit",
        "rate-limit",
        "too many requests",
        "tokens per minute",
        "output tokens per minute",
        "otpm",
        "tpd",
        "tpm",
    ]

    return any(marker in text for marker in rate_limit_markers)


def _get_retry_delay(exc: Exception, attempt: int) -> float:
    """
    Try to extract a provider-suggested retry delay.

    Example:
        Please try again in 28.44s
    """

    text = str(exc)

    patterns = [
        r"try again in\s+([0-9]+(?:\.[0-9]+)?)s",
        r"retry after\s+([0-9]+(?:\.[0-9]+)?)s",
        r"retry-after[:\s]+([0-9]+(?:\.[0-9]+)?)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            try:
                return min(
                    float(match.group(1)),
                    MAX_RETRY_DELAY,
                )

            except ValueError:
                pass

    # Exponential backoff + jitter.
    delay = INITIAL_RETRY_DELAY * (2 ** attempt)

    jitter = random.uniform(
        0,
        0.5,
    )

    return min(
        delay + jitter,
        MAX_RETRY_DELAY,
    )


def safe_invoke(model, prompt: str):
    """
    Safely invoke Groq.

    Handles:
    - 429 rate limits
    - transient provider failures
    - concurrent calls within this process

    Does NOT retry invalid 400 requests.
    """

    for attempt in range(MAX_LLM_RETRIES + 1):

        try:

            # Serialize all Groq calls.
            with _llm_execution_lock:

                result = model.invoke(prompt)

            return result

        except Exception as exc:

            if not _is_rate_limit_error(exc):

                logger.exception(
                    "LLM request failed with non-rate-limit error"
                )

                raise

            if attempt >= MAX_LLM_RETRIES:

                logger.error(
                    "Groq rate limit persisted after retries"
                )

                raise RuntimeError(
                    "The AI service is temporarily rate limited. "
                    "Please wait a few seconds and try again."
                ) from exc

            delay = _get_retry_delay(
                exc,
                attempt,
            )

            logger.warning(
                "Groq rate limit hit. "
                "Retrying in %.2f seconds "
                "(attempt %d/%d)",
                delay,
                attempt + 1,
                MAX_LLM_RETRIES,
            )

            time.sleep(delay)


# ============================================================
# HELPERS
# ============================================================

def add_trace(
    state: AgentState,
    message: str,
):
    return [
        *state.get("trace", []),
        message,
    ]


def clean_text(
    text: str,
    max_chars: int,
) -> str:

    if not text:
        return ""

    text = str(text).strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n...[context truncated]"


def build_kb_context(state: AgentState) -> str:

    docs = state.get("kb_docs", [])

    chunks = []

    for doc in docs[:MAX_KB_DOCS]:

        source = doc.metadata.get(
            "source",
            "unknown",
        )

        content = doc.page_content.strip()

        chunks.append(
            f"[Source: {source}]\n{content}"
        )

    return clean_text(
        "\n\n".join(chunks),
        MAX_KB_CONTEXT_CHARS,
    )


def build_web_context(state: AgentState) -> str:

    return clean_text(
        state.get("web_results", ""),
        MAX_WEB_CONTEXT_CHARS,
    )


# ============================================================
# ROUTING
# ============================================================

def is_casual_message(question: str) -> bool:

    q = question.lower().strip()

    casual_patterns = [
        r"^hi$",
        r"^hello$",
        r"^hey$",
        r"^hey there$",
        r"^hi there$",
        r"^good morning$",
        r"^good afternoon$",
        r"^good evening$",
        r"^thanks$",
        r"^thank you$",
        r"^thx$",
        r"^bye$",
        r"^goodbye$",
        r"^how are you\??$",
    ]

    return any(
        re.match(pattern, q)
        for pattern in casual_patterns
    )


def route_question(state: AgentState):

    question = state["question"].strip()

    if is_casual_message(question):

        route = "direct"

    else:

        # For an HR assistant, default unknown questions to the
        # private KB instead of spending an LLM call on routing.
        route = "kb"

    return {
        "source_used": route,
        "trace": add_trace(
            state,
            f"Router → {route.upper()}",
        ),
    }


def route_after_router(
    state: AgentState,
) -> Literal[
    "retrieve_kb",
    "direct_answer",
]:

    if state["source_used"] == "kb":

        return "retrieve_kb"

    return "direct_answer"


# ============================================================
# PRIVATE KB RETRIEVAL
# ============================================================

def retrieve_kb(state: AgentState):

    docs = get_retriever().invoke(
        state["current_query"]
    )

    return {
        "kb_docs": docs,
        "trace": add_trace(
            state,
            f"Private KB retrieval → {len(docs)} chunks",
        ),
    }


# ============================================================
# KB EVIDENCE GRADING
# ============================================================

def grade_kb(state: AgentState):

    docs = state.get("kb_docs", [])

    # No documents = definitely insufficient.
    if not docs:

        return {
            "kb_grade": "weak",
            "trace": add_trace(
                state,
                "KB evidence grade → WEAK (no documents)",
            ),
        }

    context = build_kb_context(state)

    response = safe_invoke(
        fast_llm(),
        f"""
You are evaluating evidence for an enterprise HR assistant.

Return EXACTLY ONE WORD:

GOOD

or

WEAK

GOOD means the supplied company HR evidence directly contains
enough information to answer the employee's question.

WEAK means the evidence is missing, unrelated, or insufficient.

Do not explain your decision.

Question:
{state["question"]}

Company HR evidence:
{context}
""",
    )

    grade_text = response.content.strip().upper()

    # Be defensive. Never depend on JSON.
    if grade_text.startswith("GOOD"):

        grade = "good"

    else:

        grade = "weak"

    return {
        "kb_grade": grade,
        "trace": add_trace(
            state,
            f"KB evidence grade → {grade.upper()}",
        ),
    }


def after_kb(
    state: AgentState,
) -> Literal[
    "generate_from_kb",
    "search_web",
]:

    if state["kb_grade"] == "good":

        return "generate_from_kb"

    return "search_web"


# ============================================================
# WEB SEARCH
# ============================================================

def search_web(state: AgentState):

    result = web_search_tool().invoke(
        {
            "query": state["current_query"]
        }
    )

    lines = []
    citations = []

    if isinstance(result, dict):

        if result.get("answer"):

            lines.append(
                "Search answer: "
                + str(result["answer"])
            )

        for item in result.get("results", []):

            title = item.get(
                "title",
                "",
            )

            url = item.get(
                "url",
                "",
            )

            content = item.get(
                "content",
                "",
            )

            lines.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content}"
            )

            if url:

                citations.append(
                    {
                        "title": title or url,
                        "url": url,
                        "type": "web",
                    }
                )

    else:

        lines.append(
            str(result)
        )

    web_context = clean_text(
        "\n\n".join(lines),
        MAX_WEB_CONTEXT_CHARS,
    )

    return {
        "web_results": web_context,
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(
            state,
            "Web fallback → Tavily search",
        ),
    }


# ============================================================
# WEB EVIDENCE GRADING
# ============================================================

def grade_web(state: AgentState):

    web_context = build_web_context(
        state
    )

    if not web_context:

        return {
            "web_grade": "weak",
            "trace": add_trace(
                state,
                "Web evidence grade → WEAK (no results)",
            ),
        }

    response = safe_invoke(
        fast_llm(),
        f"""
Evaluate whether the web evidence is sufficient to answer
the question.

Return EXACTLY ONE WORD:

GOOD

or

WEAK

Do not explain.

Question:
{state["question"]}

Web evidence:
{web_context}
""",
    )

    grade_text = response.content.strip().upper()

    if grade_text.startswith("GOOD"):

        grade = "good"

    else:

        grade = "weak"

    return {
        "web_grade": grade,
        "trace": add_trace(
            state,
            f"Web evidence grade → {grade.upper()}",
        ),
    }


# ============================================================
# QUERY REWRITE
# ============================================================

def rewrite_query(state: AgentState):

    response = safe_invoke(
        fast_llm(),
        f"""
Rewrite this employee-support question into a concise search query.

Add useful HR/policy keywords.

Do NOT answer the question.

Return ONLY the rewritten query.

Question:
{state["question"]}
""",
    )

    rewritten = response.content.strip()

    rewritten = rewritten[:500]

    if not rewritten:

        rewritten = state["question"]

    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(
            state,
            f"Query rewrite → {rewritten}",
        ),
    }


def after_web(
    state: AgentState,
) -> Literal[
    "generate_from_web",
    "rewrite_query",
    "insufficient",
]:

    if state["web_grade"] == "good":

        return "generate_from_web"

    # Only one rewrite attempt.
    if state["retry_count"] < 1:

        return "rewrite_query"

    return "insufficient"


# ============================================================
# ANSWER FROM PRIVATE KB
# ============================================================

def generate_from_kb(state: AgentState):

    context = build_kb_context(
        state
    )

    response = safe_invoke(
        llm(),
        f"""
You are an enterprise HR policy and employee support copilot.

Answer the employee's question using ONLY the private company
HR knowledge base provided below.

Rules:

1. Do not invent company policies.
2. Do not assume information that is not present.
3. Be concise and practical.
4. If there are steps, use a numbered list.
5. Clearly distinguish policy requirements from general advice.
6. If the evidence does not answer something, say so.
7. Do not mention external web sources.
8. State that the answer is based on the company's private
   knowledge base.

Employee question:
{state["question"]}

Private company HR knowledge base:
{context}
""",
    )

    answer = response.content.strip()

    citations = []

    seen = set()

    for doc in state.get(
        "kb_docs",
        [],
    ):

        source = doc.metadata.get(
            "source",
            "Private KB",
        )

        if source not in seen:

            seen.add(source)

            citations.append(
                {
                    "title": source.split("/")[-1],
                    "url": "",
                    "type": "private_kb",
                }
            )

    return {
        "answer": answer,
        "source_used": "private_kb",
        "citations": citations,
        "trace": add_trace(
            state,
            "Answer generation → PRIVATE KB",
        ),
    }


# ============================================================
# ANSWER FROM WEB
# ============================================================

def generate_from_web(state: AgentState):

    context = build_web_context(
        state
    )

    response = safe_invoke(
        llm(),
        f"""
You are an enterprise HR policy and employee support copilot.

The company's private HR knowledge base did not contain enough
evidence for this question.

Answer ONLY from the external web evidence below.

IMPORTANT:

- Clearly state that this is external public information.
- Do not present external information as company policy.
- Explain that HR validation may be required.
- Do not invent information.
- Be concise and practical.

Employee question:
{state["question"]}

External web evidence:
{context}
""",
    )

    return {
        "answer": response.content.strip(),
        "source_used": "web_search",
        "trace": add_trace(
            state,
            "Answer generation → WEB SEARCH",
        ),
    }


# ============================================================
# DIRECT ANSWER
# ============================================================

def direct_answer(state: AgentState):

    response = safe_invoke(
        fast_llm(),
        f"""
Respond briefly and naturally to this casual message.

Do not discuss HR policies.

Message:
{state["question"]}
""",
    )

    return {
        "answer": response.content.strip(),
        "source_used": "direct",
        "trace": add_trace(
            state,
            "Direct response → no retrieval",
        ),
    }


# ============================================================
# INSUFFICIENT EVIDENCE
# ============================================================

def insufficient(state: AgentState):

    return {
        "answer": (
            "I couldn't find enough reliable evidence in the "
            "company HR knowledge base or external search to "
            "answer confidently. Please contact the HR team or "
            "provide more details."
        ),
        "source_used": "insufficient_evidence",
        "trace": add_trace(
            state,
            "Stopped → insufficient reliable evidence",
        ),
    }


# ============================================================
# LANGGRAPH
# ============================================================

def build_graph():

    graph = StateGraph(
        AgentState
    )

    graph.add_node(
        "route_question",
        route_question,
    )

    graph.add_node(
        "retrieve_kb",
        retrieve_kb,
    )

    graph.add_node(
        "grade_kb",
        grade_kb,
    )

    graph.add_node(
        "search_web",
        search_web,
    )

    graph.add_node(
        "grade_web",
        grade_web,
    )

    graph.add_node(
        "rewrite_query",
        rewrite_query,
    )

    graph.add_node(
        "generate_from_kb",
        generate_from_kb,
    )

    graph.add_node(
        "generate_from_web",
        generate_from_web,
    )

    graph.add_node(
        "direct_answer",
        direct_answer,
    )

    graph.add_node(
        "insufficient",
        insufficient,
    )

    # START
    graph.add_edge(
        START,
        "route_question",
    )

    # ROUTER
    graph.add_conditional_edges(
        "route_question",
        route_after_router,
        {
            "retrieve_kb": "retrieve_kb",
            "direct_answer": "direct_answer",
        },
    )

    # KB
    graph.add_edge(
        "retrieve_kb",
        "grade_kb",
    )

    graph.add_conditional_edges(
        "grade_kb",
        after_kb,
        {
            "generate_from_kb": "generate_from_kb",
            "search_web": "search_web",
        },
    )

    # WEB
    graph.add_edge(
        "search_web",
        "grade_web",
    )

    graph.add_conditional_edges(
        "grade_web",
        after_web,
        {
            "generate_from_web": "generate_from_web",
            "rewrite_query": "rewrite_query",
            "insufficient": "insufficient",
        },
    )

    # REWRITE
    graph.add_edge(
        "rewrite_query",
        "retrieve_kb",
    )

    # TERMINAL NODES
    graph.add_edge(
        "generate_from_kb",
        END,
    )

    graph.add_edge(
        "generate_from_web",
        END,
    )

    graph.add_edge(
        "direct_answer",
        END,
    )

    graph.add_edge(
        "insufficient",
        END,
    )

    return graph.compile()


agent_graph = build_graph()


# ============================================================
# PUBLIC API
# ============================================================

def ask(question: str):

    question = (question or "").strip()

    if not question:

        raise ValueError(
            "Question cannot be empty."
        )

    # Prevent enormous user prompts.
    question = question[:4000]

    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }

    return agent_graph.invoke(
        initial
    )
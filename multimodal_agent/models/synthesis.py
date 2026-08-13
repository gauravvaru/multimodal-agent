"""Structured final synthesis output schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SummaryResponse(BaseModel):
    """Summarization output schema."""

    one_line_summary: str
    bullets: list[str] = Field(min_length=3, max_length=3)
    sentences: list[str] = Field(min_length=5, max_length=5)


class SentimentResponse(BaseModel):
    """Sentiment analysis output schema."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str


class CodeExplanationResponse(BaseModel):
    """Code explanation output schema."""

    language: str
    explanation: str
    bugs_issues: list[str] = Field(default_factory=list)
    time_complexity: str
    space_complexity: str


class ComparisonResponse(BaseModel):
    """Cross-input comparison output schema."""

    conclusion: str
    similarities: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class RagAnswerResponse(BaseModel):
    """Grounded RAG answer output schema."""

    answer: str
    insufficient_evidence: bool = False
    citations: list[str] = Field(default_factory=list)


class GroundedAnswerResponse(BaseModel):
    """Generic grounded answer output schema."""

    answer: str


def format_summary_response(response: SummaryResponse) -> str:
    bullets = "\n".join(f"- {item}" for item in response.bullets)
    paragraph = " ".join(response.sentences)
    return f"{response.one_line_summary}\n\n{bullets}\n\n{paragraph}"


def format_sentiment_response(response: SentimentResponse) -> str:
    return (
        f"Label: {response.label}\n"
        f"Confidence: {response.confidence:.2f}\n"
        f"Justification: {response.justification}"
    )


def format_code_explanation_response(response: CodeExplanationResponse) -> str:
    issues = "\n".join(f"- {item}" for item in response.bugs_issues) or "- None identified"
    return (
        f"Language: {response.language}\n\n"
        f"Explanation:\n{response.explanation}\n\n"
        f"Bugs/Issues:\n{issues}\n\n"
        f"Time complexity: {response.time_complexity}\n"
        f"Space complexity: {response.space_complexity}"
    )


def format_comparison_response(response: ComparisonResponse) -> str:
    def _section(title: str, items: list[str]) -> str:
        if not items:
            return f"{title}:\n- None identified"
        return f"{title}:\n" + "\n".join(f"- {item}" for item in items)

    sections = [
        f"Conclusion: {response.conclusion}",
        _section("Similarities", response.similarities),
        _section("Differences", response.differences),
        _section("Contradictions", response.contradictions),
        _section("Evidence", response.evidence),
    ]
    return "\n\n".join(sections)


def format_rag_answer_response(response: RagAnswerResponse) -> str:
    if response.insufficient_evidence:
        return "I could not find sufficient evidence to answer the question."
    
    citations = "\n".join(f"- {item}" for item in response.citations)
    return f"{response.answer}\n\nCitations:\n{citations}"


def format_grounded_answer_response(response: GroundedAnswerResponse) -> str:
    return response.answer
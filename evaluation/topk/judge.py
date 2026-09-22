"""Pointwise relevance judging: would a correct answer use THIS chunk?

The metric this experiment turns on is a count — "how many of the k retrieved
chunks are worth sending?" — so the judge is the simplest thing that yields a
count: one binary verdict per (question, chunk) pair, with a short reason kept
so a surprising number can be audited rather than trusted.

Three choices, each of which would change the answer if made differently:

**Pointwise, not listwise.** Every chunk is judged alone, in its own request.
Showing the judge all k chunks at once is ~3x cheaper and introduces position
bias — a chunk's verdict starts depending on its neighbours, which is the very
thing being varied between k=3 and k=11. The cheap version cannot measure the
expensive question.

**Binary, not graded.** LLM judges are dependable on "is this usable" and noisy
on "rate this 1-5". A count needs no more resolution than that.

**"Usable", not "on-topic".** ``RUBRIC`` below is the whole experiment. Under an
on-topic rubric a formal-verification syllabus scores relevant for "who works on
formal verification" — it is not, it cannot name a person. That is precisely the
failure the per-namespace split was built around (``shared/config.py``), so a
loose rubric would hide the thing worth measuring.

The absolute counts this produces are soft: a different rubric moves them all.
The *differences between k* are not, because every k is scored over the same
judged chunks, so a rubric bias is a constant that cancels. ``report.py``'s
decision rule keys on the differences for exactly that reason.

Three API constraints on Claude Fable 5.1, each a hard 400 rather than a
degraded result, which together dictate the call shape below:

1. ``temperature`` is rejected — determinism is not available as a knob.
2. ``thinking`` is rejected in any explicit form; it is always on. Depth is
   controlled with ``output_config.effort`` instead.
3. Forced ``tool_choice`` (``any``/``tool``) is rejected, so the structured
   verdict comes from ``output_config.format`` — a schema, not a forced tool.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from shared.retry import with_backoff

#: The judge. Named rather than defaulted through ``core.llm`` because this is
#: not the serving path: the generator answers questions, this scores retrieval,
#: and pinning them together would make a model change to one silently re-score
#: the other.
DEFAULT_JUDGE_MODEL = "claude-fable-5-1"

#: Classification over one short document. Effort is also the dominant cost
#: lever here, because thinking bills as output — see ``README.md`` for the
#: measured comparison against ``medium`` that justifies this default.
DEFAULT_EFFORT = "low"

#: Bumped whenever ``RUBRIC`` changes. It is part of the cache key, so an edited
#: rubric re-judges instead of silently mixing verdicts from two definitions of
#: "relevant" into one mean.
RUBRIC_VERSION = "v1"

#: Thinking is always on and bills as output, so this needs real headroom.
#: Unused headroom is free; hitting the cap truncates the JSON and costs a call.
MAX_TOKENS = 4096

RUBRIC = """\
You are judging retrieval quality for a question-answering system about \
Northeastern University people and courses.

You will be shown ONE question and ONE retrieved chunk of text.

Decide: would a correct, cited answer to this question actually draw on this \
chunk?

  relevant = true   The chunk contains information the answer would use or \
cite — a person who matches, a course that matches, a fact the question asks \
for.
  relevant = false  The chunk is about the same subject area but supplies \
nothing the answer needs.

Rules:
- Topical similarity is NOT enough. A course description about a topic does not \
answer "who works on this topic" — it names no person. A person's biography \
does not answer "which course covers this" — it names no course.
- Judge this chunk on its own. Do not assume other chunks were retrieved, and \
do not reward a chunk for being a reasonable near-miss.
- If the question asks for something the corpus plainly cannot supply (course \
prices, private contact details, an unrelated discipline), then NO chunk is \
relevant. Say false.

Give a reason of at most 20 words."""


class Relevance(BaseModel):
    """The judge's structured verdict."""

    relevant: bool = Field(description="Would a correct cited answer use this chunk?")
    reason: str = Field(description="At most 20 words.")


@dataclass(frozen=True)
class Verdict:
    """One judged (question, chunk) pair, plus what it cost to learn.

    Token counts are carried so the run can report *measured* spend rather than
    an estimate; a cache hit reports zero, because it is.
    """

    relevant: bool
    reason: str
    cached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


def is_retryable(error: BaseException) -> bool:
    """Whether the judge should back off and try again.

    Matches on the SDK's typed exceptions rather than message text. Imported
    lazily so this module stays importable — and unit-testable — on a machine
    with no ``anthropic`` installed.
    """
    try:
        import anthropic
    except ImportError:  # pragma: no cover - only on a machine without the SDK
        return False
    if isinstance(error, (anthropic.RateLimitError, anthropic.APIConnectionError)):
        return True
    return isinstance(error, anthropic.APIStatusError) and error.status_code >= 500


class JudgeRefusal(RuntimeError):
    """The judge declined the pair outright.

    Its own class because a refusal must never be silently recorded as
    ``relevant=False`` — that would look like a retrieval miss and quietly bias
    every k downward. The runner records it as an error and prints the count.
    """


class RelevanceJudge:
    """Claude Fable 5.1, scoring one chunk at a time, with a disk cache.

    The cache is keyed on everything that could change a verdict — rubric
    version, model, effort, question, chunk id, and the chunk text itself — so
    re-analysis is free while any real change re-judges. That matters more than
    it looks: the same run is replayed once per k value in ``report.py``, and
    iterating on the report format would otherwise re-buy every verdict.
    """

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        effort: str = DEFAULT_EFFORT,
        cache_dir: Path | None = None,
        client=None,
        max_attempts: int = 5,
    ):
        self.model = model
        self.effort = effort
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = client
        self.max_attempts = max_attempts

    @property
    def client(self):
        """Lazily build the client from ANTHROPIC_API_KEY.

        ``max_retries=0`` on purpose: ``with_backoff`` is this project's single
        retry policy, and layering the SDK's on top of it would multiply the
        attempts and the wall-clock without anyone choosing to.
        """
        if self._client is None:
            import anthropic

            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set; the judge cannot run")
            self._client = anthropic.Anthropic(api_key=key, max_retries=0)
        return self._client

    # -- the judgement ------------------------------------------------------
    def judge(self, question: str, chunk_id: str, text: str) -> Verdict:
        """Score one (question, chunk) pair, reading the cache first."""
        cached = self._read(question, chunk_id, text)
        if cached is not None:
            return cached
        verdict = with_backoff(
            lambda: self._ask(question, text),
            is_retryable=is_retryable,
            max_attempts=self.max_attempts,
            label=f"judge {chunk_id}",
        )
        self._write(question, chunk_id, text, verdict)
        return verdict

    def _ask(self, question: str, text: str) -> Verdict:
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=RUBRIC,
            output_format=Relevance,
            output_config={"effort": self.effort},
            messages=[
                {"role": "user", "content": f"QUESTION\n{question}\n\nCHUNK\n{text}"}
            ],
        )
        if response.stop_reason == "refusal":
            raise JudgeRefusal(f"judge declined: {getattr(response.stop_details, 'category', None)}")

        parsed = next(
            (b.parsed_output for b in response.content if getattr(b, "parsed_output", None)),
            None,
        )
        if parsed is None:
            # Reached when the schema came back unfilled — almost always a
            # max_tokens truncation. Raised rather than defaulted to False for
            # the same reason as JudgeRefusal.
            raise RuntimeError(f"judge returned no parsed verdict (stop={response.stop_reason})")

        return Verdict(
            relevant=bool(parsed.relevant),
            reason=str(parsed.reason),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    # -- the cache ----------------------------------------------------------
    def _path(self, question: str, chunk_id: str, text: str) -> Path | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(
            "\x00".join([RUBRIC_VERSION, self.model, self.effort, question, chunk_id, text]).encode()
        ).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read(self, question: str, chunk_id: str, text: str) -> Verdict | None:
        """A cached verdict, or None. An unreadable entry is a miss, not an error."""
        path = self._path(question, chunk_id, text)
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError, KeyError):
            return None
        return Verdict(relevant=bool(payload["relevant"]), reason=payload["reason"], cached=True)

    def _write(self, question: str, chunk_id: str, text: str, verdict: Verdict) -> None:
        path = self._path(question, chunk_id, text)
        if path is None:
            return
        # Via a temporary file so an interrupted run cannot leave a half-written
        # entry that the next run has to treat as a miss.
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"relevant": verdict.relevant, "reason": verdict.reason}, ensure_ascii=False)
        )
        temporary.replace(path)

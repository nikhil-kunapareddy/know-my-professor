"""The judge: the model that scores, and the embeddings it scores with.

The only module here that needs credentials, and the only one that imports a
provider SDK at module scope — import it and you are committing to the ``eval``
extra being installed. The CLI imports it inside ``main`` for exactly that
reason, so ``--help`` works without it.

Two deliberate choices:

**The judge is not built through ``core.llm``.** Every judged metric asks for a
*structured* verdict (a list of extracted claims, a verdict per claim), which
DeepEval gets by passing a Pydantic schema to the model. ``Generator.generate``
returns prose and cannot express that, so wrapping it would mean
re-implementing the schema coercion inside this package. The judge therefore
talks to DeepEval's own ``AnthropicModel``, while still reading the same
``ANTHROPIC_API_KEY`` the service does.

**Judge embeddings are the corpus embeddings.** The one metric that needs
vectors (``semantic_similarity``) goes through ``shared.embeddings``, so
similarity is measured in the same space the index was built in. A second
embedding model here would make that score incomparable to retrieval's, for no
gain.

Three findings from wiring this up, each a silent or confusing failure if
forgotten — hence the code below rather than a note in a README:

1. **Never send ``temperature``.** The Anthropic SDK's ``Messages.create()`` no
   longer accepts sampling parameters, so any judge call carrying one dies with
   ``TypeError: AsyncMessages.create() got an unexpected keyword argument
   'temperature'``. ``AnthropicModel`` only forwards it when it is explicitly
   configured — but it also picks one up from DeepEval's own ``TEMPERATURE``
   setting, which is an env var any machine might have set. ``build_judge``
   therefore asserts it came out unset rather than trusting the default.
2. **A metric with no ``model=`` is an OpenAI metric.** DeepEval resolves a
   missing or string-valued model to ``OpenAIModel`` and then fails on a
   missing ``OPENAI_API_KEY``, from deep inside the metric rather than at
   construction. There is no OpenAI key in this project by design;
   ``metrics.build_metric`` refuses rather than letting that happen.
3. **There is no judge cache on this path.** DeepEval caches for
   ``deepeval test run``, not for metrics driven directly, so re-judging an
   unchanged run would re-buy every verdict. ``CachingJudge`` below adds one.

G-Eval scores through log probabilities when the judge is an OpenAI model and
falls back to plain schema extraction otherwise, which is the path taken here:
Anthropic returns no log probabilities. The fallback is supported and tested
upstream; the practical effect is that ``answer_correctness`` lands on coarser
values than it would with a GPT judge. It is not comparable across providers —
but then neither is any judged metric.
"""

from __future__ import annotations

import hashlib
import json
import typing as t
from pathlib import Path

from deepeval.models import AnthropicModel
from deepeval.models.base_model import DeepEvalBaseLLM

from shared.config import EMBED_RATE_LIMIT_SLEEP_SECONDS
from shared.embeddings import build_embedder
from shared.embeddings.base import Embedder
from shared.settings import require_env

#: Judging is a classification job over short text, not a reasoning job, and it
#: runs dozens of times per case. Override with --judge-model when comparing a
#: stronger judge; expect the absolute numbers to shift when you do, so do not
#: compare across judges.
DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"

#: Only Anthropic is wired up. DeepEval also ships OpenAI, Gemini, Bedrock and
#: others, but this project holds an Anthropic key and no OpenAI key; naming the
#: provider keeps the CLI honest about what it does and does not support.
JUDGE_PROVIDERS = ("anthropic",)
DEFAULT_JUDGE_PROVIDER = "anthropic"


def build_judge(
    model: str | None = None,
    provider: str = DEFAULT_JUDGE_PROVIDER,
    cache_dir: str | None = None,
) -> DeepEvalBaseLLM:
    """Build the judge model, optionally wrapped in a disk cache."""
    if provider not in JUDGE_PROVIDERS:
        raise ValueError(
            f"unknown judge provider {provider!r}; available: {JUDGE_PROVIDERS}"
        )

    judge = AnthropicModel(
        model=model or DEFAULT_JUDGE_MODEL,
        api_key=require_env("ANTHROPIC_API_KEY"),
    )
    assert_no_temperature(judge)
    return CachingJudge(judge, Path(cache_dir)) if cache_dir else judge


def assert_no_temperature(judge: t.Any) -> t.Any:
    """Refuse a judge configured with a sampling parameter the SDK rejects.

    Separate from ``build_judge`` so a test can prove it without needing a key:
    if DeepEval starts defaulting ``temperature`` again, or the machine has
    ``TEMPERATURE`` set in its environment, this fails once at construction
    instead of on every judged call, several minutes and a lot of quota later.
    """
    temperature = getattr(judge, "temperature", None)
    if temperature is not None:
        raise ValueError(
            f"the judge was configured with temperature={temperature!r}, which the "
            "Anthropic Messages API rejects outright — every judged call would fail. "
            "Unset the TEMPERATURE environment variable, or stop passing it."
        )
    return judge


def build_judge_embedder(embedder: Embedder | None = None) -> Embedder:
    """The corpus embedder, paced for the free tier.

    Mistral's free tier reports ``x-ratelimit-limit-req-minute: 60``, and
    ``semantic_similarity`` costs one batched request per case on top of the
    query embed retrieval already spent. ``pace_seconds`` is the knob ingest
    already uses for exactly this reason (see ``shared/embeddings/mistral.py``);
    serving still passes 0 — this tax is the evaluation's alone.
    """
    return embedder or build_embedder(pace_seconds=EMBED_RATE_LIMIT_SLEEP_SECONDS)


class CachingJudge(DeepEvalBaseLLM):
    """A judge that remembers its verdicts on disk.

    Keyed on the model name, the prompt, and the schema the caller asked for —
    so a changed prompt or a changed metric still re-judges, while re-running an
    unchanged evaluation costs nothing. That matters more here than it looks:
    iterating on the report format, the gates, or a single metric otherwise
    means re-buying every verdict in the run.

    A cache hit reports a cost of 0.0, because it is one.
    """

    def __init__(self, judge: DeepEvalBaseLLM, cache_dir: Path):
        self.judge = judge
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        super().__init__(judge.get_model_name())

    # -- DeepEvalBaseLLM ---------------------------------------------------
    def load_model(self) -> t.Any:
        return self.judge

    def get_model_name(self) -> str:
        return self.judge.get_model_name()

    @property
    def temperature(self) -> t.Any:
        """Surfaced so ``assert_no_temperature`` sees through the wrapper."""
        return getattr(self.judge, "temperature", None)

    def generate(self, prompt: str, schema: t.Any = None, *args: t.Any, **kwargs: t.Any):
        cached = self._read(prompt, schema)
        if cached is not None:
            return cached, 0.0
        result, cost = self.judge.generate(prompt, schema, *args, **kwargs)
        self._write(prompt, schema, result)
        return result, cost

    async def a_generate(self, prompt: str, schema: t.Any = None, *args: t.Any, **kwargs: t.Any):
        cached = self._read(prompt, schema)
        if cached is not None:
            return cached, 0.0
        result, cost = await self.judge.a_generate(prompt, schema, *args, **kwargs)
        self._write(prompt, schema, result)
        return result, cost

    # -- the cache itself --------------------------------------------------
    def _path(self, prompt: str, schema: t.Any) -> Path:
        digest = hashlib.sha256(
            "\x00".join(
                [self.get_model_name(), getattr(schema, "__name__", ""), prompt]
            ).encode()
        ).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read(self, prompt: str, schema: t.Any) -> t.Any | None:
        """Return a cached verdict, or None on a miss or an unreadable entry.

        A corrupt or stale entry is a miss, never an error: a half-written file
        from an interrupted run must not be able to fail an evaluation that
        would otherwise succeed by simply asking the judge again.
        """
        path = self._path(prompt, schema)
        if not path.exists():
            self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text())["payload"]
            value = schema.model_validate(payload) if schema is not None else payload
        except Exception:  # noqa: BLE001 - any unreadable entry is simply a miss
            self.misses += 1
            return None
        self.hits += 1
        return value

    def _write(self, prompt: str, schema: t.Any, result: t.Any) -> None:
        """Store a verdict, giving up silently if it cannot be serialised."""
        payload = result.model_dump() if hasattr(result, "model_dump") else result
        try:
            serialised = json.dumps(
                {"model": self.get_model_name(), "payload": payload}, ensure_ascii=False
            )
        except (TypeError, ValueError):
            return
        path = self._path(prompt, schema)
        # Written via a temporary file so an interrupted run cannot leave a
        # half-written entry that the next run would have to treat as a miss.
        temporary = path.with_suffix(".tmp")
        temporary.write_text(serialised)
        temporary.replace(path)

    def describe_cache(self) -> str:
        return f"{self.hits} hit(s), {self.misses} miss(es) in {self.cache_dir}"

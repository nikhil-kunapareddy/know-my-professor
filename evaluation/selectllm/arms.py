"""The candidate configurations under test, and what each one costs.

An arm is a (model, effort) pair rather than a model, because effort changes
both answer quality and — since thinking bills as output — the price of a
request. Comparing bare model names would confound the two.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.llm.anthropic import supports_effort


@dataclass(frozen=True)
class Price:
    """Anthropic first-party rates, $/MTok. Cached 2026-06-24."""

    input: float
    output: float


#: Model id -> rates. Ids are exact: never append a date suffix.
PRICING: dict[str, Price] = {
    "claude-opus-5": Price(5.0, 25.0),
    "claude-sonnet-5": Price(2.0, 10.0),
    "claude-haiku-4-5": Price(1.0, 5.0),
    "claude-fable-5-1": Price(10.0, 50.0),
}


@dataclass(frozen=True)
class Arm:
    """One deployable configuration.

    ``effort`` is ``None`` where the model does not accept the parameter. That
    is not a default — it is the only legal value for Claude Haiku 4.5, which
    returns a hard 400 on ``effort``.
    """

    model: str
    effort: str | None

    @property
    def price(self) -> Price:
        return PRICING[self.model]

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Dollars for one call.

        ``output_tokens`` must be the provider's billed figure, which includes
        thinking tokens. Deriving cost from visible answer length instead
        understates a high-effort arm roughly twofold.
        """
        return (input_tokens * self.price.input + output_tokens * self.price.output) / 1e6


#: The four arms. opus5-low is the incumbent and the baseline.
#:
#: Two comparisons are clean by construction and one is not:
#:   opus5-med vs sonnet5-med  -- model, at matched effort
#:   opus5-low vs opus5-med    -- effort, at matched model
#:   haiku45                   -- unmatched; it cannot take an effort setting,
#:                                so it serves as the cost floor rather than as
#:                                a controlled comparison.
ARMS: dict[str, Arm] = {
    "opus5-low": Arm("claude-opus-5", "low"),
    "opus5-med": Arm("claude-opus-5", "medium"),
    "sonnet5-med": Arm("claude-sonnet-5", "medium"),
    "haiku45": Arm("claude-haiku-4-5", None),
}

#: The judge. Deliberately NOT an arm: it cannot rank its own answers.
JUDGE_MODEL = "claude-fable-5-1"
JUDGE_EFFORT = "low"


def _validate() -> None:
    """Fail at import on an arm whose effort the model would silently drop.

    ``AnthropicGenerator`` drops an unsupported ``effort`` rather than raising,
    so an arm named ``haiku45-high`` would run with no effort at all while every
    table in the report claimed otherwise. A mislabelled arm is worse than a
    crash: it produces a plausible number for a configuration that never ran.
    """
    for name, arm in ARMS.items():
        if arm.model not in PRICING:
            raise ValueError(f"arm {name!r}: no price for {arm.model!r}")
        if arm.effort is not None and not supports_effort(arm.model):
            raise ValueError(
                f"arm {name!r} sets effort={arm.effort!r} but {arm.model!r} rejects it; "
                "the parameter would be dropped and the arm mislabelled"
            )
    if JUDGE_MODEL in {a.model for a in ARMS.values()}:
        raise ValueError(f"judge {JUDGE_MODEL!r} is also an arm; it would grade itself")


_validate()

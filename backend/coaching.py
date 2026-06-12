"""
Coaching feedback generation via the Anthropic Claude API.

This module is independently unit-testable: the Anthropic client is injected
as a dependency rather than constructed internally, so tests can pass a mock
without touching the network.

Expected ``scores`` dict structure::

    {
        "technique": "roundhouse_kick",
        "metric_deltas": {
            "hip_rotation_deg": {"user": 15.2, "reference": 55.4, "delta": -40.2},
            "chamber_height_ratio": {"user": 0.72, "reference": 0.91, "delta": -0.19},
            ...
        },
        "keyframe_descriptions": [
            "Chamber: knee raised to hip height; guard hands relaxed",
            "Impact: hip barely rotated; striking leg not fully extended",
        ],
    }
"""

import os
from typing import Union

import anthropic

# Allow override via env var; default to a capable, cost-effective model.
MODEL: str = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

SYSTEM_PROMPT: str = (
    "You are an expert martial arts coach specialising in biomechanical analysis. "
    "You receive quantitative data from a pose-estimation scoring pipeline and write "
    "concise, grounded coaching feedback. "
    "Your rules:\n"
    "  1. Reference the specific numeric deltas provided — never give generic advice.\n"
    "  2. Explain *why* each gap matters for power, balance, or technique.\n"
    "  3. Prioritise the largest deviations.\n"
    "  4. Write in second person ('your hip…'), plain prose, 2–4 sentences per criterion.\n"
    "  5. No bullet points or headers — continuous paragraphs only."
)


def build_prompt(scores: dict) -> str:
    """Construct the user-turn prompt from a scores dict.

    The prompt embeds every metric-delta value so Claude can ground its
    feedback in real numbers rather than generic cues.

    Args:
        scores: Dict with ``technique``, ``metric_deltas``, and
                ``keyframe_descriptions`` keys (see module docstring).

    Returns:
        A plain-text prompt string ready to send as the ``user`` message.
    """
    technique: str = scores.get("technique", "unknown_technique")
    metric_deltas: dict = scores.get("metric_deltas", {})
    keyframe_descriptions: list = scores.get("keyframe_descriptions", [])

    lines = [
        f"Technique: {technique.replace('_', ' ').title()}",
        "",
        "Metric deltas (user value vs. expert reference):",
    ]

    for metric, values in metric_deltas.items():
        user_val = values.get("user", "N/A")
        ref_val = values.get("reference", "N/A")
        delta = values.get("delta", "N/A")
        label = metric.replace("_", " ")

        if isinstance(user_val, float) and isinstance(ref_val, float) and isinstance(delta, float):
            lines.append(
                f"  {label}: user={user_val:.2f}, reference={ref_val:.2f}, delta={delta:+.2f}"
            )
        else:
            lines.append(
                f"  {label}: user={user_val}, reference={ref_val}, delta={delta}"
            )

    if keyframe_descriptions:
        lines.append("")
        lines.append("Annotated keyframes:")
        for desc in keyframe_descriptions:
            lines.append(f"  - {desc}")

    lines.append("")
    lines.append(
        "Write grounded coaching feedback that references the specific numeric deltas "
        "above. Focus on the largest deviations first. Do not give generic advice."
    )

    return "\n".join(lines)


def generate_feedback(scores: dict, client: anthropic.Anthropic) -> str:
    """Generate plain-language coaching feedback for a technique attempt.

    Args:
        scores: Dict with ``technique``, ``metric_deltas``, and
                ``keyframe_descriptions`` (see module docstring).
        client: An already-instantiated :class:`anthropic.Anthropic` client.
                Injected rather than created here so callers can supply a
                mock during testing.

    Returns:
        Coaching feedback as a plain-language string.

    Raises:
        anthropic.APIError: Propagated if the Claude API call fails.
    """
    prompt = build_prompt(scores)

    message = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    if not message.content:
        raise ValueError(
            f"Anthropic API returned an empty content list "
            f"(stop_reason={message.stop_reason!r})"
        )
    return message.content[0].text

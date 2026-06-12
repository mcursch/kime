"""
Unit tests for backend/coaching.py.

All tests mock the Anthropic client so they run without a real API key
or network access.

Acceptance criterion verified here:
  "A unit test mocking the Anthropic client verifies that coaching.py
   passes metric-delta values into the prompt string."
"""

from unittest.mock import MagicMock

import anthropic
import pytest

from backend.coaching import MODEL, SYSTEM_PROMPT, build_prompt, generate_feedback

# ---------------------------------------------------------------------------
# Sample data shared across tests
# ---------------------------------------------------------------------------

SAMPLE_SCORES = {
    "technique": "roundhouse_kick",
    "metric_deltas": {
        "hip_rotation_deg": {"user": 15.2, "reference": 55.4, "delta": -40.2},
        "chamber_height_ratio": {"user": 0.72, "reference": 0.91, "delta": -0.19},
        "extension_angle_deg": {"user": 148.0, "reference": 165.0, "delta": -17.0},
        "guard_drop_deg": {"user": 22.0, "reference": 5.0, "delta": 17.0},
    },
    "keyframe_descriptions": [
        "Chamber: knee raised to hip height; guard hands relaxed at sides",
        "Impact: hip barely rotated; striking leg not fully extended",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client(feedback_text: str = "mock feedback") -> MagicMock:
    """Return a MagicMock shaped like anthropic.Anthropic."""
    content_block = MagicMock()
    content_block.text = feedback_text

    message = MagicMock()
    message.content = [content_block]

    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.return_value = message
    return client


def _extract_user_prompt(mock_client: MagicMock) -> str:
    """Pull the user-turn content string out of the messages.create call."""
    call = mock_client.messages.create.call_args
    # Keyword argument 'messages' is a list of role/content dicts.
    messages = call.kwargs.get("messages", [])
    return "".join(
        msg["content"] for msg in messages if isinstance(msg, dict) and msg.get("role") == "user"
    )


# ---------------------------------------------------------------------------
# build_prompt tests
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_includes_technique_name(self):
        prompt = build_prompt(SAMPLE_SCORES)
        assert "Roundhouse Kick" in prompt

    def test_includes_user_value_for_every_metric(self):
        prompt = build_prompt(SAMPLE_SCORES)
        assert "15.20" in prompt  # hip_rotation_deg user
        assert "0.72" in prompt   # chamber_height_ratio user
        assert "148.00" in prompt  # extension_angle_deg user
        assert "22.00" in prompt  # guard_drop_deg user

    def test_includes_reference_value_for_every_metric(self):
        prompt = build_prompt(SAMPLE_SCORES)
        assert "55.40" in prompt  # hip_rotation_deg reference
        assert "0.91" in prompt   # chamber_height_ratio reference
        assert "165.00" in prompt  # extension_angle_deg reference
        assert "5.00" in prompt   # guard_drop_deg reference

    def test_includes_delta_value_for_every_metric(self):
        prompt = build_prompt(SAMPLE_SCORES)
        assert "-40.20" in prompt  # hip_rotation_deg delta
        assert "-0.19" in prompt   # chamber_height_ratio delta
        assert "-17.00" in prompt  # extension_angle_deg delta
        assert "+17.00" in prompt  # guard_drop_deg delta (positive)

    def test_includes_keyframe_descriptions(self):
        prompt = build_prompt(SAMPLE_SCORES)
        assert "Chamber: knee raised to hip height" in prompt
        assert "Impact: hip barely rotated" in prompt

    def test_empty_metric_deltas(self):
        scores = {
            "technique": "front_kick",
            "metric_deltas": {},
            "keyframe_descriptions": [],
        }
        prompt = build_prompt(scores)
        assert "Front Kick" in prompt

    def test_missing_keyframe_descriptions_key(self):
        scores = {
            "technique": "straight_punch",
            "metric_deltas": {"extension_angle_deg": {"user": 160.0, "reference": 175.0, "delta": -15.0}},
        }
        prompt = build_prompt(scores)
        assert "Straight Punch" in prompt
        assert "160.00" in prompt


# ---------------------------------------------------------------------------
# generate_feedback tests  (acceptance criterion)
# ---------------------------------------------------------------------------


class TestGenerateFeedback:
    def test_returns_text_from_api(self):
        client = _make_mock_client("Your hip rotation is 40° short of the reference.")
        result = generate_feedback(SAMPLE_SCORES, client)
        assert result == "Your hip rotation is 40° short of the reference."

    def test_calls_messages_create_once(self):
        client = _make_mock_client()
        generate_feedback(SAMPLE_SCORES, client)
        assert client.messages.create.call_count == 1

    def test_passes_correct_model(self):
        client = _make_mock_client()
        generate_feedback(SAMPLE_SCORES, client)
        model_arg = client.messages.create.call_args.kwargs.get("model")
        assert model_arg == MODEL
        assert "claude" in model_arg.lower()

    def test_passes_system_prompt(self):
        client = _make_mock_client()
        generate_feedback(SAMPLE_SCORES, client)
        system_arg = client.messages.create.call_args.kwargs.get("system")
        assert system_arg == SYSTEM_PROMPT

    # ---------------------------------------------------------------------- #
    # KEY ACCEPTANCE CRITERION:                                               #
    # metric-delta values must appear verbatim in the prompt sent to Claude.  #
    # ---------------------------------------------------------------------- #

    def test_prompt_contains_user_hip_rotation_value(self):
        """hip_rotation_deg user value (15.2 → '15.20') is in the prompt."""
        client = _make_mock_client()
        generate_feedback(SAMPLE_SCORES, client)
        prompt = _extract_user_prompt(client)
        assert "15.20" in prompt, f"Expected '15.20' in prompt:\n{prompt}"

    def test_prompt_contains_reference_hip_rotation_value(self):
        """hip_rotation_deg reference value (55.4 → '55.40') is in the prompt."""
        client = _make_mock_client()
        generate_feedback(SAMPLE_SCORES, client)
        prompt = _extract_user_prompt(client)
        assert "55.40" in prompt, f"Expected '55.40' in prompt:\n{prompt}"

    def test_prompt_contains_hip_rotation_delta(self):
        """hip_rotation_deg delta (-40.2 → '-40.20') is in the prompt."""
        client = _make_mock_client()
        generate_feedback(SAMPLE_SCORES, client)
        prompt = _extract_user_prompt(client)
        assert "-40.20" in prompt, f"Expected '-40.20' in prompt:\n{prompt}"

    def test_prompt_contains_all_metric_delta_values(self):
        """Every user/reference/delta value for every metric appears in the prompt."""
        client = _make_mock_client()
        generate_feedback(SAMPLE_SCORES, client)
        prompt = _extract_user_prompt(client)

        expected_substrings = [
            # hip_rotation_deg
            "15.20", "55.40", "-40.20",
            # chamber_height_ratio
            "0.72", "0.91", "-0.19",
            # extension_angle_deg
            "148.00", "165.00", "-17.00",
            # guard_drop_deg
            "22.00", "5.00", "+17.00",
        ]
        missing = [s for s in expected_substrings if s not in prompt]
        assert not missing, (
            f"The following metric values were absent from the prompt: {missing}\n"
            f"Prompt:\n{prompt}"
        )

    def test_prompt_contains_technique_name(self):
        client = _make_mock_client()
        generate_feedback(SAMPLE_SCORES, client)
        prompt = _extract_user_prompt(client)
        assert "Roundhouse Kick" in prompt

    def test_prompt_contains_keyframe_descriptions(self):
        client = _make_mock_client()
        generate_feedback(SAMPLE_SCORES, client)
        prompt = _extract_user_prompt(client)
        assert "Chamber: knee raised to hip height" in prompt
        assert "Impact: hip barely rotated" in prompt

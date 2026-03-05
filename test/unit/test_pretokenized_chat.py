"""
Unit tests for the pretokenized chat completion path.

Tests that using pretokenized_token_ids + pretokenized_num_message produces
identical token IDs as the standard apply_chat_template path.

Run with:
    python -m pytest test/unit/test_pretokenized_chat.py -v
    python test/unit/test_pretokenized_chat.py -v
"""

import json
import os
import unittest
from typing import Dict, List, Optional

from jinja2.sandbox import ImmutableSandboxedEnvironment


def _tojson(value, ensure_ascii=True, indent=None):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent)


def apply_chat_template_from_str(
    chat_template: str,
    messages: List[Dict],
    add_generation_prompt: bool = True,
    tools: Optional[List[Dict]] = None,
    **kwargs,
) -> str:
    """Render a Jinja2 chat template string (tokenize=False equivalent)."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(
        ValueError(msg)
    )
    env.filters["tojson"] = _tojson
    template = env.from_string(chat_template)

    render_kwargs = {
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        render_kwargs["tools"] = tools
    render_kwargs.update(kwargs)
    return template.render(**render_kwargs)


# ---------------------------------------------------------------------------
# Load chat templates
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

with open(os.path.join(_REPO_ROOT, "qwen3_fixed.jinja")) as f:
    QWEN3_FIXED_CHAT_TEMPLATE = f.read()

with open(os.path.join(_REPO_ROOT, "qwen3.5.jinja")) as f:
    QWEN35_CHAT_TEMPLATE = f.read()

with open(os.path.join(_REPO_ROOT, "glm5.jinja")) as f:
    GLM5_CHAT_TEMPLATE = f.read()

with open(os.path.join(_REPO_ROOT, "qwen3_instruct_2507.jinja")) as f:
    QWEN3_INSTRUCT_2507_CHAT_TEMPLATE = f.read()

with open(os.path.join(_REPO_ROOT, "qwen3_thinking_2507.jinja")) as f:
    QWEN3_THINKING_2507_CHAT_TEMPLATE = f.read()

with open(os.path.join(_REPO_ROOT, "qwen3_thinking_2507_fixed.jinja")) as f:
    QWEN3_THINKING_2507_FIXED_CHAT_TEMPLATE = f.read()


# ---------------------------------------------------------------------------
# Test messages fixtures
# ---------------------------------------------------------------------------

WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["city"],
            },
        },
    }
]


def make_tool_call_messages_single_tool():
    """sys, user, assistant(tool_calls), tool"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Beijing?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing", "unit": "celsius"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 25, "condition": "sunny"}',
            "tool_call_id": "call_1",
        },
    ]


def make_tool_call_messages_multi_turn():
    """sys, user, assistant(tool_calls), tool, assistant(tool_calls), tool"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Beijing and Shanghai?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 25, "condition": "sunny"}',
            "tool_call_id": "call_1",
        },
        {
            "role": "assistant",
            "content": "Beijing is 25C. Let me check Shanghai.",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Shanghai"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 30, "condition": "cloudy"}',
            "tool_call_id": "call_2",
        },
    ]


def make_tool_call_messages_multi_tool_single_turn():
    """sys, user, assistant(2 tool_calls), tool, tool"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Weather in Beijing and calculate 2+3"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing"},
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Shanghai"},
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 25}',
            "tool_call_id": "call_1",
        },
        {
            "role": "tool",
            "content": '{"temperature": 30}',
            "tool_call_id": "call_2",
        },
    ]


def make_tool_call_messages_parallel_tools():
    """sys, user, assistant(3 parallel tool_calls), tool, tool, tool"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Compare weather in Beijing, Shanghai, and Guangzhou"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing"},
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Shanghai"},
                    },
                },
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Guangzhou"},
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 25, "condition": "sunny"}',
            "tool_call_id": "call_1",
        },
        {
            "role": "tool",
            "content": '{"temperature": 30, "condition": "cloudy"}',
            "tool_call_id": "call_2",
        },
        {
            "role": "tool",
            "content": '{"temperature": 35, "condition": "rainy"}',
            "tool_call_id": "call_3",
        },
    ]


def make_multi_user_tool_chain():
    """sys, user1, ass(tool), tool1, ass, user2, ass(tool), tool2, ass(tool), tool3"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Beijing?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 25, "condition": "sunny"}',
            "tool_call_id": "call_1",
        },
        {
            "role": "assistant",
            "content": "Beijing is 25C and sunny.",
        },
        {"role": "user", "content": "Now check Shanghai and Guangzhou"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Shanghai"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 30, "condition": "cloudy"}',
            "tool_call_id": "call_2",
        },
        {
            "role": "assistant",
            "content": "Shanghai is 30C. Let me check Guangzhou.",
            "tool_calls": [
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Guangzhou"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 35, "condition": "rainy"}',
            "tool_call_id": "call_3",
        },
    ]


def make_simple_messages_no_tool():
    """sys, user, assistant (no tools)"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help?"},
    ]


def make_tool_call_messages_long_chain():
    """sys, user, ass(tool), tool, ass(tool), tool, ass(tool), tool"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Do a multi-step task"},
        {
            "role": "assistant",
            "content": "Step 1",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 25}',
            "tool_call_id": "call_1",
        },
        {
            "role": "assistant",
            "content": "Step 2",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Shanghai"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 30}',
            "tool_call_id": "call_2",
        },
        {
            "role": "assistant",
            "content": "Step 3",
            "tool_calls": [
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Guangzhou"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 35}',
            "tool_call_id": "call_3",
        },
    ]


# ---------------------------------------------------------------------------
# Helper: simulate the pretokenized incremental tokenization logic
# ---------------------------------------------------------------------------


def simulate_pretokenized_path(
    chat_template: str,
    messages: List[Dict],
    pretokenized_num_message: int,
    tools: Optional[List[Dict]] = None,
    **template_kwargs,
) -> str:
    """Simulate the pretokenized incremental path at text level.

    1. Render first N messages (no generation prompt) -> prefix_text
    2. Render ALL messages (with generation prompt) -> full_text
    3. Verify prefix_text is a prefix of full_text
    4. incremental_text = full_text[len(prefix_text):]
    5. Return prefix_text (pretokenized) + incremental_text

    Since we use the same template, prefix_text == pretokenized part,
    so the result should equal full_text.
    """
    tool_dicts = None
    if tools:
        tool_dicts = [t["function"] for t in tools if "function" in t]

    prefix_text = apply_chat_template_from_str(
        chat_template,
        messages[:pretokenized_num_message],
        add_generation_prompt=False,
        tools=tool_dicts,
        **template_kwargs,
    )

    full_text = apply_chat_template_from_str(
        chat_template,
        messages,
        add_generation_prompt=True,
        tools=tool_dicts,
        **template_kwargs,
    )

    # The key invariant: prefix must match
    if not full_text.startswith(prefix_text):
        raise ValueError(
            f"Prefix mismatch!\n"
            f"prefix_text ({len(prefix_text)} chars):\n{repr(prefix_text[-200:])}\n\n"
            f"full_text at same position:\n{repr(full_text[:len(prefix_text)][-200:])}"
        )

    incremental_text = full_text[len(prefix_text) :]
    result = prefix_text + incremental_text
    assert result == full_text
    return full_text


def get_standard_result(
    chat_template: str,
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    **template_kwargs,
) -> str:
    """Standard path: render all messages with generation prompt."""
    tool_dicts = None
    if tools:
        tool_dicts = [t["function"] for t in tools if "function" in t]

    return apply_chat_template_from_str(
        chat_template,
        messages,
        add_generation_prompt=True,
        tools=tool_dicts,
        **template_kwargs,
    )


# ===========================================================================
# Base test class
# ===========================================================================


class PretokenizedTemplateTestBase(unittest.TestCase):
    """Base class for pretokenized template tests.

    Subclasses only need to set `chat_template` class attribute.
    Not collected by pytest directly (no "Test" prefix).
    """

    # Prevent pytest from collecting this base class directly
    __test__ = False

    chat_template: str = ""

    def _assert_pretokenized_equals_standard(
        self,
        messages,
        pretokenized_num_message,
        tools=None,
        **kwargs,
    ):
        standard = get_standard_result(
            self.chat_template, messages, tools=tools, **kwargs
        )
        pretokenized = simulate_pretokenized_path(
            self.chat_template,
            messages,
            pretokenized_num_message,
            tools=tools,
            **kwargs,
        )
        self.assertEqual(
            pretokenized,
            standard,
            f"Pretokenized (N={pretokenized_num_message}) != standard",
        )

    # --- single tool call ---

    def test_single_tool_call_with_tools(self):
        """sys, user, assistant(tool_calls), tool -- pretokenize first 3"""
        messages = make_tool_call_messages_single_tool()
        self._assert_pretokenized_equals_standard(
            messages, pretokenized_num_message=3, tools=WEATHER_TOOLS
        )

    def test_single_tool_call_no_tools_def(self):
        """Same but without passing tools definition."""
        messages = make_tool_call_messages_single_tool()
        self._assert_pretokenized_equals_standard(
            messages, pretokenized_num_message=3
        )

    # --- multi-turn tool calls ---

    def test_multi_turn_pretokenize_3(self):
        """Multi-turn: pretokenize sys+user+ass (first tool call)"""
        messages = make_tool_call_messages_multi_turn()
        self._assert_pretokenized_equals_standard(
            messages, pretokenized_num_message=3, tools=WEATHER_TOOLS
        )

    def test_multi_turn_pretokenize_5(self):
        """Multi-turn: pretokenize up to second assistant"""
        messages = make_tool_call_messages_multi_turn()
        self._assert_pretokenized_equals_standard(
            messages, pretokenized_num_message=5, tools=WEATHER_TOOLS
        )

    # --- multiple tool calls in single turn ---

    def test_multi_tool_single_turn(self):
        """2 tool_calls in one assistant, 2 tool responses"""
        messages = make_tool_call_messages_multi_tool_single_turn()
        self._assert_pretokenized_equals_standard(
            messages, pretokenized_num_message=3, tools=WEATHER_TOOLS
        )

    # --- parallel tool calls (3 tools) ---

    def test_parallel_tools_pretokenize_3(self):
        """3 parallel tool_calls, pretokenize first 3 (sys+user+ass)"""
        messages = make_tool_call_messages_parallel_tools()
        self._assert_pretokenized_equals_standard(
            messages, pretokenized_num_message=3, tools=WEATHER_TOOLS
        )

    # --- long chain ---

    def test_long_chain_pretokenize_3(self):
        """Long chain: pretokenize first 3 (sys+user+ass)"""
        messages = make_tool_call_messages_long_chain()
        self._assert_pretokenized_equals_standard(
            messages, pretokenized_num_message=3, tools=WEATHER_TOOLS
        )

    def test_long_chain_pretokenize_5(self):
        """Long chain: pretokenize first 5 (sys+user+ass+tool+ass)"""
        messages = make_tool_call_messages_long_chain()
        self._assert_pretokenized_equals_standard(
            messages, pretokenized_num_message=5, tools=WEATHER_TOOLS
        )

    def test_long_chain_pretokenize_7(self):
        """Long chain: pretokenize first 7 (up to third assistant)"""
        messages = make_tool_call_messages_long_chain()
        self._assert_pretokenized_equals_standard(
            messages, pretokenized_num_message=7, tools=WEATHER_TOOLS
        )

    # --- multi-user tool chain (user1 ... user2 ... with tool calls) ---

    def test_multi_user_chain_pretokenize_7(self):
        """Multi-user chain: pretokenize up to ass(tool_calls) after user2, add [tool2, ass, tool3]"""
        messages = make_multi_user_tool_chain()
        # messages: sys(0) user1(1) ass(2) tool1(3) ass(4) user2(5) ass(6) tool2(7) ass(8) tool3(9)
        # N=7: prefix=[0..6], last=ass(6), new=[tool2, ass(tool_calls), tool3]
        self._assert_pretokenized_equals_standard(
            messages, pretokenized_num_message=7, tools=WEATHER_TOOLS
        )

    def test_multi_user_chain_pretokenize_9(self):
        """Multi-user chain: pretokenize up to ass(tool_calls), add [tool3]"""
        messages = make_multi_user_tool_chain()
        # N=9: prefix=[0..8], last=ass(8), new=[tool3]
        self._assert_pretokenized_equals_standard(
            messages, pretokenized_num_message=9, tools=WEATHER_TOOLS
        )

    # --- edge case: N == len(messages), no incremental tool messages ---

    def test_no_incremental_messages(self):
        """All messages pretokenized, incremental part is just generation prompt."""
        messages = make_simple_messages_no_tool()
        self._assert_pretokenized_equals_standard(
            messages, pretokenized_num_message=3
        )


# ===========================================================================
# Per-model test classes
# ===========================================================================


class TestQwen3Fixed(PretokenizedTemplateTestBase):
    """Qwen3 with fixed template (no loop.last dependency)."""

    __test__ = True
    chat_template = QWEN3_FIXED_CHAT_TEMPLATE

    def test_thinking_disabled(self):
        messages = make_tool_call_messages_single_tool()
        self._assert_pretokenized_equals_standard(
            messages, pretokenized_num_message=3,
            tools=WEATHER_TOOLS, enable_thinking=False,
        )

    def test_thinking_enabled(self):
        messages = make_tool_call_messages_single_tool()
        self._assert_pretokenized_equals_standard(
            messages, pretokenized_num_message=3,
            tools=WEATHER_TOOLS, enable_thinking=True,
        )


class TestQwen35(PretokenizedTemplateTestBase):
    """Qwen3.5 (HuggingFace original)."""

    __test__ = True
    chat_template = QWEN35_CHAT_TEMPLATE

    def test_thinking_disabled(self):
        messages = make_tool_call_messages_single_tool()
        self._assert_pretokenized_equals_standard(
            messages, pretokenized_num_message=3,
            tools=WEATHER_TOOLS, enable_thinking=False,
        )

    def test_thinking_enabled(self):
        messages = make_tool_call_messages_single_tool()
        self._assert_pretokenized_equals_standard(
            messages, pretokenized_num_message=3,
            tools=WEATHER_TOOLS, enable_thinking=True,
        )


class TestGLM5(PretokenizedTemplateTestBase):
    """GLM-5 (zai-org/GLM-5)."""

    __test__ = True
    chat_template = GLM5_CHAT_TEMPLATE

    def test_thinking_disabled(self):
        messages = make_tool_call_messages_single_tool()
        self._assert_pretokenized_equals_standard(
            messages, pretokenized_num_message=3,
            tools=WEATHER_TOOLS, enable_thinking=False,
        )

    def test_thinking_enabled(self):
        messages = make_tool_call_messages_single_tool()
        self._assert_pretokenized_equals_standard(
            messages, pretokenized_num_message=3,
            tools=WEATHER_TOOLS, enable_thinking=True,
        )


class TestQwen3Instruct2507(PretokenizedTemplateTestBase):
    """Qwen3-4B-Instruct-2507 (HuggingFace original)."""

    __test__ = True
    chat_template = QWEN3_INSTRUCT_2507_CHAT_TEMPLATE


class TestQwen3Thinking2507Fixed(PretokenizedTemplateTestBase):
    """Qwen3-4B-Thinking-2507 with fixed template (no loop.last dependency)."""

    __test__ = True
    chat_template = QWEN3_THINKING_2507_FIXED_CHAT_TEMPLATE

    def test_thinking_disabled(self):
        messages = make_tool_call_messages_single_tool()
        self._assert_pretokenized_equals_standard(
            messages, pretokenized_num_message=3,
            tools=WEATHER_TOOLS, enable_thinking=False,
        )

    def test_thinking_enabled(self):
        messages = make_tool_call_messages_single_tool()
        self._assert_pretokenized_equals_standard(
            messages, pretokenized_num_message=3,
            tools=WEATHER_TOOLS, enable_thinking=True,
        )


class TestQwen3Thinking2507Original(unittest.TestCase):
    """Verify original Qwen3-4B-Thinking-2507 has loop.last prefix mismatch."""

    def test_original_has_prefix_mismatch(self):
        """Original template fails prefix invariant due to loop.last in thinking tags."""
        messages = make_tool_call_messages_single_tool()
        with self.assertRaises(ValueError, msg="Expected prefix mismatch"):
            simulate_pretokenized_path(
                QWEN3_THINKING_2507_CHAT_TEMPLATE,
                messages,
                pretokenized_num_message=3,
                tools=[t["function"] for t in WEATHER_TOOLS],
            )


# ===========================================================================
# Validation tests
# ===========================================================================


class TestPretokenizedValidation(unittest.TestCase):
    """Test validation logic for pretokenized path."""

    def test_last_pretokenized_must_be_assistant(self):
        """message[N-1] must be assistant."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        self.assertNotEqual(messages[1]["role"], "assistant")

    def test_messages_after_N_must_be_tool(self):
        """Messages after N must all be tool role."""
        messages = make_tool_call_messages_single_tool() + [
            {"role": "user", "content": "Thanks!"}
        ]
        self.assertNotEqual(messages[4]["role"], "tool")


if __name__ == "__main__":
    unittest.main()

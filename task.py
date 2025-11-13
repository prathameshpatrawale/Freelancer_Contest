"""
task.py — Text Cleaning & Normalization RL Task
-----------------------------------------------

This file defines:
  • The RL-task prompt
  • The built-in tool definitions (python_expression + submit_answer)
  • Tool handlers
  • The grading function

The task teaches a realistic ML-engineering skill:
**cleaning and normalizing unstructured text data** — one of the most common
steps in any NLP pipeline.

The model must:
  1. Read the messy list of user reviews provided in the prompt.
  2. Use the python_expression tool to write/run Python that:
       - lowercases text
       - removes HTML tags
       - normalizes whitespace
       - removes toxic reviews
       - deduplicates the dataset
  3. Submit the final cleaned list via submit_answer.

Multiple valid solution paths are allowed.
The grader checks logical correctness (not formatting), so creativity is fine.
"""

from typing import Any, TypedDict, Callable
from anthropic.types import ToolUnionParam
import re


# ---------------------------------------------------------------------
# Dataset used inside prompt + grader
# ---------------------------------------------------------------------
ORIGINAL_REVIEWS = [
    "I LOVE this product!!!  ",
    "<br>Terrible idea... absolutely STUPID design<br>",
    "Decent, could be better",
    "I love this product!!!",
    "<div>What a nonsense feature</div>",
    "Pretty good overall",
    "This is an IDIOT move",
    "Pretty good overall  ",
    "Excellent — highly recommend!",
]

TOXIC_WORDS = {"idiot", "stupid", "nonsense"}  # case-insensitive whole-word match


# ---------------------------------------------------------------------
# Tool result types
# ---------------------------------------------------------------------
class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


# ---------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------
def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """
    Executes Python code written by the agent.
    stdout is captured and returned.

    The agent should use this tool to load/clean the reviews using Python.
    """
    try:
        local_ns = {}
        from io import StringIO
        import sys

        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            exec(expression, local_ns, local_ns)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        return {"result": output, "error": None}

    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """
    Tool used by the agent to submit the final cleaned list.
    Whatever value is sent here is passed to the grader.
    """
    return {"answer": answer, "submitted": True}


# ---------------------------------------------------------------------
# Anthropic tool definitions
# ---------------------------------------------------------------------
TOOLS: list[ToolUnionParam] = [
    {
        "name": "python_expression",
        "description": "Execute Python code. Use print(...) to show intermediate results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Python code to execute via exec().",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "submit_answer",
        "description": "Submit the final cleaned list.",
        "input_schema": {
            "type": "object",
            "properties": {"answer": {"description": "Final cleaned list"}},
            "required": ["answer"],
        },
    },
]

TOOL_HANDLERS: dict[str, Callable[..., Any]] = {
    "python_expression": python_expression_tool,
    "submit_answer": submit_answer_tool,
}


# ---------------------------------------------------------------------
# PROMPT (the RL task)
# ---------------------------------------------------------------------
PROMPT = f"""
You are assisting in a real ML workflow. A small text dataset needs cleaning
before it can be used to train a sentiment classifier.

Below is the dataset, stored in the Python variable `reviews`:

reviews = {ORIGINAL_REVIEWS}

Your task is to clean these reviews using Python. You MUST use the
`python_expression` tool to run your Python code. After computing the final
cleaned list, call the `submit_answer` tool with the cleaned list.

Apply all of the following cleaning rules:

1. Convert all text to lowercase.
2. Remove HTML tags (anything between '<' and '>').
3. Collapse repeated whitespace into a single space, and trim leading/trailing whitespace.
4. Remove any review that contains one or more toxic words
   (case-insensitive whole words): {sorted(list(TOXIC_WORDS))}.
5. Remove duplicate reviews (keep only one copy).
6. Do NOT invent new reviews — every returned item must originate from the
   original dataset after cleaning.

Submission format:
  • Use python_expression to compute a Python list of cleaned strings.
  • Then call submit_answer with that list.
  • Do not output anything else as your final submission.

Notes:
  • Order does not matter.
  • Multiple correct solutions exist.
  • Be careful not to hallucinate rows or partially clean text.

Now use python_expression to perform the cleaning, and then submit the result.
"""


# ---------------------------------------------------------------------
# Grading function
# ---------------------------------------------------------------------
def _canonical_clean(text: str) -> str | None:
    """
    Deterministic cleaning function used by the grader to produce
    the correct cleaned set.

    Performs:
      - lowercase
      - strip HTML
      - normalize whitespace
      - remove toxic content
    """
    if not isinstance(text, str):
        return None

    # lowercase
    s = text.lower()

    # remove HTML tags
    s = re.sub(r"<.*?>", "", s)

    # normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # remove if toxic words appear as whole tokens
    tokens = re.findall(r"\w+", s)
    if any(tok in TOXIC_WORDS for tok in tokens):
        return None

    return s if s else None


def grading_func(result: Any) -> bool:
    """
    Validate the agent's submission.

    Requirements:
      • result must be a list of strings
      • no HTML tags
      • all lowercase
      • no toxic content
      • no duplicates
      • every submitted row must match a canonical cleaned row
      • submitted set must match canonical set exactly
    """
    if not isinstance(result, list):
        return False

    # compute canonical cleaned set
    canonical = set()
    for r in ORIGINAL_REVIEWS:
        cleaned = _canonical_clean(r)
        if cleaned is not None:
            canonical.add(cleaned)

    seen = set()

    for item in result:
        if not isinstance(item, str):
            return False

        # no angle brackets allowed
        if "<" in item or ">" in item:
            return False

        # must be lowercase
        if item != item.lower():
            return False

        # normalize whitespace for comparison
        norm = re.sub(r"\s+", " ", item).strip()

        # check toxicity
        tokens = re.findall(r"\w+", norm)
        if any(tok in TOXIC_WORDS for tok in tokens):
            return False

        # no duplicates in submission
        if norm in seen:
            return False
        seen.add(norm)

        # must come from canonical set
        if norm not in canonical:
            return False

    # final check: exact match of sets
    return seen == canonical

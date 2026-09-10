from typing import Literal

from inspect_ai.log import transcript
from inspect_ai.model import GenerateConfig, Model, ResponseSchema, get_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState
from inspect_ai.util import json_schema
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_baselines.evals.hle_tools_v0.prompts import EQUALITY_CHECKER_PROMPT

DEFAULT_JUDGE_MODEL = "openai/gpt-5.6-luna"
DEFAULT_JUDGE_REASONING_EFFORT = "medium"


class EqualityJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int = Field(ge=0, le=100)


def parse_judgment(text: str) -> EqualityJudgment:
    return EqualityJudgment.model_validate_json(text)


@scorer(metrics=[accuracy(), stderr()])
def hle_scorer(
    judge_model: str | Model = DEFAULT_JUDGE_MODEL,
    judge_reasoning_effort: str = DEFAULT_JUDGE_REASONING_EFFORT,
) -> Scorer:
    """Score the complete model response with the Artificial Analysis HLE judge."""

    async def score(state: TaskState, target: Target) -> Score:
        judge = get_model(judge_model)
        prompt = EQUALITY_CHECKER_PROMPT.format(
            question=state.input_text,
            response=state.output.completion,
            correct_answer=target.text.strip(),
        )
        max_tokens = 4096
        for attempt in range(1, 4):
            output = await judge.generate(
                input=prompt,
                config=GenerateConfig(
                    max_tokens=max_tokens,
                    reasoning_effort=judge_reasoning_effort,
                    response_schema=ResponseSchema(
                        name="hle_equality_judgment",
                        description="Structured result from the HLE equality checker.",
                        json_schema=json_schema(EqualityJudgment),
                        strict=True,
                    ),
                ),
            )
            # Retain even unparseable judge responses in the sample's Inspect events.
            transcript().info(
                {
                    "attempt": attempt,
                    "max_tokens": max_tokens,
                    "output": output.model_dump(mode="json"),
                },
                source="hle_judge_attempt",
            )
            try:
                judgment = parse_judgment(output.completion)
                break
            except ValidationError as error:
                cap_exhausted = (
                    max_tokens == 4096
                    and output.usage is not None
                    and output.usage.output_tokens == 4096
                    and any(
                        choice.stop_reason == "max_tokens" for choice in output.choices
                    )
                    and any(
                        item["type"] == "json_invalid"
                        and "EOF" in item.get("ctx", {}).get("error", "")
                        for item in error.errors()
                    )
                )
                if attempt == 3:
                    raise
                if cap_exhausted:
                    max_tokens = 16384
        return Score(
            value=CORRECT if judgment.correct == "yes" else INCORRECT,
            answer=judgment.extracted_final_answer,
            explanation=judgment.reasoning,
            metadata={
                "confidence": judgment.confidence / 100,
                "answer_type": state.metadata.get("answer_type", "exact_match"),
                "judge_model": judge.name,
                "judge_reasoning_effort": judge_reasoning_effort,
                "judge_attempts": attempt,
                "judge_max_tokens": max_tokens,
            },
        )

    return score

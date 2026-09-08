from typing import Literal

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
from pydantic import BaseModel, ConfigDict, Field

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
        output = await judge.generate(
            input=EQUALITY_CHECKER_PROMPT.format(
                question=state.input_text,
                response=state.output.completion,
                correct_answer=target.text.strip(),
            ),
            config=GenerateConfig(
                max_tokens=4096,
                reasoning_effort=judge_reasoning_effort,
                response_schema=ResponseSchema(
                    name="hle_equality_judgment",
                    description="Structured result from the HLE equality checker.",
                    json_schema=json_schema(EqualityJudgment),
                    strict=True,
                ),
            ),
        )
        judgment = parse_judgment(output.completion)
        return Score(
            value=CORRECT if judgment.correct == "yes" else INCORRECT,
            answer=judgment.extracted_final_answer,
            explanation=judgment.reasoning,
            metadata={
                "confidence": judgment.confidence / 100,
                "answer_type": state.metadata.get("answer_type", "exact_match"),
                "judge_model": judge.name,
                "judge_reasoning_effort": judge_reasoning_effort,
            },
        )

    return score

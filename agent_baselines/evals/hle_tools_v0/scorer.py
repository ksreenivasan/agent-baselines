import json
import re
from typing import Any

from inspect_ai.model import get_model
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

JUDGE_PROMPT = """Judge whether the submitted answer matches the reference answer. Do not solve the
question and do not demand wording that is absent from the reference. Return only JSON:
{{"correct": "yes" or "no", "reasoning": "brief equivalence explanation"}}

Question: {question}
Submitted answer: {answer}
Reference answer: {target}
"""


def parse_submission(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    if set(payload) - {"answer", "confidence", "explanation"}:
        raise ValueError("submission has unexpected keys")
    answer = payload.get("answer")
    confidence = payload.get("confidence")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("submission answer is empty")
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        raise ValueError("confidence must be in [0, 1]")
    return {
        "answer": answer.strip(),
        "confidence": float(confidence),
        "explanation": str(payload.get("explanation", "")),
    }


def _choice(value: str) -> str:
    match = re.search(r"\b([A-Z])\b", value.upper())
    return match.group(1) if match else value.strip().casefold()


def _judge_json(text: str) -> dict[str, str]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("judge did not return JSON")
    payload = json.loads(text[start : end + 1])
    if payload.get("correct") not in {"yes", "no"}:
        raise ValueError("judge correctness is invalid")
    return payload


@scorer(metrics=[accuracy(), stderr()])
def hle_scorer(judge_model: str = "openai/o3-mini-2025-01-31") -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        try:
            submission = parse_submission(state.output.completion)
        except (ValueError, json.JSONDecodeError) as error:
            return Score(value=INCORRECT, explanation=f"invalid submission: {error}")

        answer = submission["answer"]
        reference = target.text.strip()
        answer_type = state.metadata.get("answer_type", "exact_match")
        judge_reasoning = ""

        if answer_type == "multiple_choice":
            correct = _choice(answer) == _choice(reference)
            judge_reasoning = "deterministic multiple-choice comparison"
        elif answer.casefold() == reference.casefold():
            correct = True
            judge_reasoning = "deterministic normalized exact match"
        else:
            question = state.input_text
            output = await get_model(judge_model).generate(
                input=JUDGE_PROMPT.format(question=question, answer=answer, target=reference)
            )
            judgment = _judge_json(output.completion)
            correct = judgment["correct"] == "yes"
            judge_reasoning = str(judgment.get("reasoning", ""))

        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=answer,
            explanation=judge_reasoning,
            metadata={"confidence": submission["confidence"], "answer_type": answer_type},
        )

    return score

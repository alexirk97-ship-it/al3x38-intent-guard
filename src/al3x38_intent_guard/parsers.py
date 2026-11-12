import json
from typing import Any, Dict, List, Optional

from .core import StepTelemetry, ToolInvocation


def parse_openai_payload(
    session_id: str,
    prompt: str,
    payload: Dict[str, Any],
    approved: bool = False,
) -> StepTelemetry:
    """Извлекает текст, все вызовы функций и logprobs из ответа формата OpenAI / vLLM / LM Studio."""
    choices = payload.get("choices", [])
    if not choices:
        return StepTelemetry(session_id=session_id, prompt=prompt, response="", approved_by_human=approved)

    choice = choices[0]
    msg = choice.get("message", {})
    text = msg.get("content") or ""

    invocations: List[ToolInvocation] = []
    raw_calls = msg.get("tool_calls")
    if isinstance(raw_calls, list):
        for call in raw_calls:
            fn = call.get("function", {})
            name = fn.get("name")
            if name:
                invocations.append(ToolInvocation(name=name, arguments=str(fn.get("arguments", ""))))

    logits: Optional[List[float]] = None
    lp = choice.get("logprobs")
    if isinstance(lp, dict) and lp.get("content"):
        first_token_info = lp["content"][0]
        top_probs = first_token_info.get("top_logprobs", [])
        if top_probs:
            logits = [float(item["logprob"]) for item in top_probs if "logprob" in item]

    return StepTelemetry(
        session_id=session_id,
        prompt=prompt,
        response=text,
        tools=invocations,
        logits=logits,
        approved_by_human=approved,
    )


def parse_ollama_payload(
    session_id: str,
    prompt: str,
    payload: Dict[str, Any],
    approved: bool = False,
) -> StepTelemetry:
    """Извлекает текст и вызовы инструментов из ответа локального сервера Ollama (/api/chat)."""
    msg = payload.get("message", {})
    text = msg.get("content") or ""

    invocations: List[ToolInvocation] = []
    raw_calls = msg.get("tool_calls")
    if isinstance(raw_calls, list):
        for call in raw_calls:
            fn = call.get("function", {})
            name = fn.get("name")
            if name:
                args_raw = fn.get("arguments", {})
                args_str = json.dumps(args_raw, ensure_ascii=False) if isinstance(args_raw, dict) else str(args_raw)
                invocations.append(ToolInvocation(name=name, arguments=args_str))

    return StepTelemetry(
        session_id=session_id,
        prompt=prompt,
        response=text,
        tools=invocations,
        logits=None,
        approved_by_human=approved,
    )

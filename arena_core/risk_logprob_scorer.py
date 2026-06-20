from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import chess

from arena_core.reranker import CandidateScore

RISK_LABEL_PREFIX = '{"risk":"'
RISK_LABELS = ("blunder", "mistake", "playable", "good")
RISK_SCORE = {"blunder": 0.0, "mistake": 1.0, "playable": 2.0, "good": 3.0}


@dataclass(frozen=True)
class RiskLogprobScorerConfig:
    adapter_dir: str
    max_seq_length: int = 1024
    min_safe_score: float = 1.0
    disable_thinking: bool = True


class RiskLogprobScorer:
    name = "risk_logprob"

    def __init__(self, config: RiskLogprobScorerConfig) -> None:
        self.config = config
        try:
            self._unsloth = import_module("unsloth")
            self._torch = import_module("torch")
        except ImportError as exc:
            raise RuntimeError(
                "ARENA_RERANKER_SCORER=risk_logprob requires the training environment "
                'dependencies. Activate .venv-train and install `pip install -e ".[train]"`.'
            ) from exc
        self._model, self._tokenizer = self._unsloth.FastModel.from_pretrained(
            model_name=config.adapter_dir,
            max_seq_length=config.max_seq_length,
            load_in_4bit=True,
        )
        self._model.eval()

    async def score(self, *, board: chess.Board, move: chess.Move) -> CandidateScore:
        prompt = build_risk_prompt(board=board, move=move)
        prompt_text = _apply_chat_template(
            tokenizer=self._tokenizer,
            messages=[{"role": "user", "content": prompt}],
            disable_thinking=self.config.disable_thinking,
        )
        risk_logprobs = _label_logprobs(
            model=self._model,
            tokenizer=self._tokenizer,
            torch=self._torch,
            prefix_text=prompt_text + RISK_LABEL_PREFIX,
            labels=list(RISK_LABELS),
        )
        risk_probs = softmax(risk_logprobs)
        expected_score = sum(
            RISK_SCORE[risk] * probability
            for risk, probability in risk_probs.items()
        )
        predicted_risk = max(risk_logprobs, key=risk_logprobs.get)
        return CandidateScore(
            centipawn_loss=None,
            classification=predicted_risk,
            vetoed=expected_score < self.config.min_safe_score,
            ranking_score=expected_score,
            details={
                "risk_probs": risk_probs,
                "risk_logprobs": risk_logprobs,
                "min_safe_score": self.config.min_safe_score,
            },
        )


def build_risk_prompt(*, board: chess.Board, move: chess.Move) -> str:
    before = board.copy(stack=False)
    candidate_san = before.san(move)
    after = before.copy(stack=False)
    after.push(move)
    return "\n".join(
        [
            "You are a chess move risk classifier.",
            "",
            "Position:",
            f"FEN before: {before.fen()}",
            f"Side to move: {'white' if before.turn == chess.WHITE else 'black'}",
            f"Candidate move: {move.uci()} ({candidate_san})",
            f"FEN after candidate: {after.fen()}",
            f"Opponent legal replies: {_legal_replies(after)}",
            "",
            "Classify whether the candidate is tactically safe.",
            "Return JSON only:",
            '{"risk":"blunder|mistake|playable|good","blunder":true,'
            '"score":0,"reason_type":"material_or_mate"}',
        ]
    )


def softmax(logprobs: dict[str, float]) -> dict[str, float]:
    max_logprob = max(logprobs.values())
    exp_values = {
        label: math.exp(logprob - max_logprob)
        for label, logprob in logprobs.items()
    }
    total = sum(exp_values.values())
    return {label: value / total for label, value in exp_values.items()}


def _legal_replies(board: chess.Board) -> str:
    return ", ".join(move.uci() for move in board.legal_moves)


def _label_logprobs(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    prefix_text: str,
    labels: list[str],
) -> dict[str, float]:
    sequences: list[list[int]] = []
    label_starts: list[int] = []
    prefix_ids = _token_ids(tokenizer=tokenizer, text=prefix_text)
    for label in labels:
        full_ids = _token_ids(tokenizer=tokenizer, text=prefix_text + label)
        label_start = _common_prefix_length(prefix_ids, full_ids)
        if label_start >= len(full_ids):
            raise ValueError(f"risk label {label!r} produced no label tokens")
        sequences.append(full_ids)
        label_starts.append(label_start)

    input_ids, attention_mask = _padded_batch(
        torch=torch,
        tokenizer=tokenizer,
        sequences=sequences,
        device=model.device,
    )
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return {
        label: _sequence_label_logprob(
            log_probs=log_probs[row_index],
            token_ids=sequences[row_index],
            label_start=label_starts[row_index],
        )
        for row_index, label in enumerate(labels)
    }


def _padded_batch(
    *,
    torch: Any,
    tokenizer: Any,
    sequences: list[list[int]],
    device: Any,
) -> tuple[Any, Any]:
    pad_token_id = _token_id(tokenizer=tokenizer, name="pad_token_id")
    if pad_token_id is None:
        pad_token_id = _token_id(tokenizer=tokenizer, name="eos_token_id")
    if pad_token_id is None:
        pad_token_id = 0
    max_length = max(len(sequence) for sequence in sequences)
    padded = [
        sequence + [pad_token_id] * (max_length - len(sequence))
        for sequence in sequences
    ]
    mask = [
        [1] * len(sequence) + [0] * (max_length - len(sequence))
        for sequence in sequences
    ]
    return (
        torch.tensor(padded, device=device),
        torch.tensor(mask, device=device),
    )


def _sequence_label_logprob(
    *,
    log_probs: Any,
    token_ids: list[int],
    label_start: int,
) -> float:
    total = 0.0
    token_count = 0
    for token_index in range(label_start, len(token_ids)):
        previous_index = token_index - 1
        if previous_index < 0:
            continue
        total += float(log_probs[previous_index, token_ids[token_index]].item())
        token_count += 1
    if token_count == 0:
        raise ValueError("risk label has no scorable tokens")
    return total / token_count


def _token_ids(*, tokenizer: Any, text: str) -> list[int]:
    try:
        tokens = tokenizer(text, add_special_tokens=False, return_tensors=None)
    except (TypeError, ValueError):
        tokens = tokenizer(text=text, add_special_tokens=False, return_tensors=None)
    ids = tokens["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    index = 0
    for left_token, right_token in zip(left, right, strict=False):
        if left_token != right_token:
            break
        index += 1
    return index


def _token_id(*, tokenizer: Any, name: str) -> int | None:
    value = getattr(tokenizer, name, None)
    if value is not None:
        return int(value)
    inner_tokenizer = getattr(tokenizer, "tokenizer", None)
    if inner_tokenizer is None:
        return None
    inner_value = getattr(inner_tokenizer, name, None)
    return int(inner_value) if inner_value is not None else None


def _apply_chat_template(
    *,
    tokenizer: Any,
    messages: list[dict[str, str]],
    disable_thinking: bool,
) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if disable_thinking:
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    enable_thinking=False,
                    **kwargs,
                )
            )
        except TypeError:
            pass
    return str(tokenizer.apply_chat_template(messages, **kwargs))

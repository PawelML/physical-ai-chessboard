import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import chess

from arena_core.evaluators.stockfish import EngineEvaluation, StockfishEvaluator
from arena_core.move_sources import MoveProposal
from arena_core.parser import parse_uci_json
from arena_core.utils import close_if_present

RERANKED_SOURCE_PREFIX = "reranked:"


class CandidateMoveSource(Protocol):
    name: str
    source_type: str

    async def propose(self, *, prompt: str, board: chess.Board) -> MoveProposal:
        """Return one raw candidate response."""


@dataclass(frozen=True)
class CandidateScore:
    centipawn_loss: int | None
    classification: str
    vetoed: bool


class RerankerScorer(Protocol):
    name: str

    async def score(self, *, board: chess.Board, move: chess.Move) -> CandidateScore:
        """Score one legal candidate move."""


@dataclass(frozen=True)
class RerankerConfig:
    n_candidates: int = 5
    veto_cpl_threshold: int = 300


@dataclass(frozen=True)
class LegalCandidate:
    move: chess.Move
    uci: str
    raw_response: str
    multiplicity: int
    first_index: int


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: LegalCandidate
    score: CandidateScore


@dataclass(frozen=True)
class RerankerSelection:
    chosen: ScoredCandidate | None
    all_vetoed: bool


class StockfishVetoScorer:
    name = "stockfish"

    def __init__(
        self,
        *,
        evaluator: StockfishEvaluator,
        veto_cpl_threshold: int = 300,
    ) -> None:
        self._evaluator = evaluator
        self._veto_cpl_threshold = veto_cpl_threshold

    async def score(self, *, board: chess.Board, move: chess.Move) -> CandidateScore:
        evaluation = self._evaluator.evaluate_move(board, move)
        return score_from_engine_evaluation(
            evaluation,
            veto_cpl_threshold=self._veto_cpl_threshold,
        )

    def close(self) -> None:
        self._evaluator.close()


class RerankedLLMMoveSource:
    source_type = "llm_reranked"

    def __init__(
        self,
        *,
        inner: CandidateMoveSource,
        scorer: RerankerScorer,
        config: RerankerConfig | None = None,
        temperature: float | None = None,
    ) -> None:
        self.inner = inner
        self.scorer = scorer
        self.config = config or RerankerConfig()
        self.temperature = temperature
        self.name = f"reranked:{inner.name}"

    async def propose(self, *, prompt: str, board: chess.Board) -> MoveProposal:
        started = time.perf_counter()
        proposals: list[MoveProposal] = []
        for _ in range(self.config.n_candidates):
            proposals.append(await self.inner.propose(prompt=prompt, board=board.copy(stack=False)))

        legal_candidates = collect_legal_candidates(
            board=board,
            raw_responses=[proposal.raw_response for proposal in proposals],
        )
        scored = [
            ScoredCandidate(
                candidate=candidate,
                score=await self.scorer.score(board=board.copy(stack=False), move=candidate.move),
            )
            for candidate in legal_candidates
        ]
        selection = select_candidate(scored)
        latency_ms = (time.perf_counter() - started) * 1000

        metadata = build_reranker_metadata(
            scorer_name=self.scorer.name,
            config=self.config,
            temperature=self.temperature,
            raw_responses=[proposal.raw_response for proposal in proposals],
            legal_candidates=legal_candidates,
            scored_candidates=scored,
            selection=selection,
        )
        first_proposal = proposals[0] if proposals else None
        if selection.chosen is None:
            if first_proposal is None:
                return MoveProposal(
                    raw_response='{"move":"0000"}',
                    latency_ms=latency_ms,
                    metadata=metadata,
                )
            return _aggregate_proposal(
                raw_response=first_proposal.raw_response,
                latency_ms=latency_ms,
                proposals=proposals,
                metadata=metadata,
            )

        return _aggregate_proposal(
            raw_response=json.dumps(
                {"move": selection.chosen.candidate.uci},
                separators=(",", ":"),
            ),
            latency_ms=latency_ms,
            proposals=proposals,
            metadata=metadata,
        )

    def close(self) -> None:
        close_if_present(self.inner)
        close_if_present(self.scorer)


def is_reranked_source_name(name: str) -> bool:
    return name.startswith(RERANKED_SOURCE_PREFIX) and bool(
        name.removeprefix(RERANKED_SOURCE_PREFIX)
    )


def inner_source_name(name: str) -> str:
    if is_reranked_source_name(name):
        return name.removeprefix(RERANKED_SOURCE_PREFIX)
    return name


def score_from_engine_evaluation(
    evaluation: EngineEvaluation,
    *,
    veto_cpl_threshold: int,
) -> CandidateScore:
    vetoed = evaluation.classification == "mate_missed"
    if evaluation.centipawn_loss is not None:
        vetoed = vetoed or evaluation.centipawn_loss > veto_cpl_threshold
    return CandidateScore(
        centipawn_loss=evaluation.centipawn_loss,
        classification=evaluation.classification,
        vetoed=vetoed,
    )


def collect_legal_candidates(
    *,
    board: chess.Board,
    raw_responses: Sequence[str],
) -> list[LegalCandidate]:
    by_uci: dict[str, LegalCandidate] = {}
    for index, raw_response in enumerate(raw_responses):
        move = _parse_legal_move(board, raw_response)
        if move is None:
            continue
        uci = move.uci()
        existing = by_uci.get(uci)
        if existing is None:
            by_uci[uci] = LegalCandidate(
                move=move,
                uci=uci,
                raw_response=raw_response,
                multiplicity=1,
                first_index=index,
            )
        else:
            by_uci[uci] = LegalCandidate(
                move=existing.move,
                uci=existing.uci,
                raw_response=existing.raw_response,
                multiplicity=existing.multiplicity + 1,
                first_index=existing.first_index,
            )
    return sorted(by_uci.values(), key=lambda candidate: candidate.first_index)


def select_candidate(candidates: Sequence[ScoredCandidate]) -> RerankerSelection:
    if not candidates:
        return RerankerSelection(chosen=None, all_vetoed=False)
    safe_candidates = [candidate for candidate in candidates if not candidate.score.vetoed]
    if safe_candidates:
        return RerankerSelection(
            chosen=min(safe_candidates, key=_safe_selection_key),
            all_vetoed=False,
        )
    return RerankerSelection(chosen=min(candidates, key=_least_bad_selection_key), all_vetoed=True)


def build_reranker_metadata(
    *,
    scorer_name: str,
    config: RerankerConfig,
    temperature: float | None,
    raw_responses: Sequence[str],
    legal_candidates: Sequence[LegalCandidate],
    scored_candidates: Sequence[ScoredCandidate],
    selection: RerankerSelection,
) -> dict[str, Any]:
    first_legal_uci = legal_candidates[0].uci if legal_candidates else None
    chosen_uci = selection.chosen.candidate.uci if selection.chosen is not None else None
    return {
        "scorer": scorer_name,
        "n_candidates_configured": config.n_candidates,
        "temperature": temperature,
        "veto_cpl_threshold": config.veto_cpl_threshold,
        "n_candidates_generated": len(raw_responses),
        "n_legal": sum(candidate.multiplicity for candidate in legal_candidates),
        "n_distinct_legal": len(legal_candidates),
        "n_vetoed": sum(1 for candidate in scored_candidates if candidate.score.vetoed),
        "veto_changed_move": (
            chosen_uci is not None and first_legal_uci is not None and chosen_uci != first_legal_uci
        ),
        "all_vetoed": selection.all_vetoed,
        "chosen_multiplicity": (
            selection.chosen.candidate.multiplicity if selection.chosen is not None else None
        ),
        "chosen_uci": chosen_uci,
        "first_legal_uci": first_legal_uci,
        "candidates": [
            {
                "uci": candidate.candidate.uci,
                "multiplicity": candidate.candidate.multiplicity,
                "centipawn_loss": candidate.score.centipawn_loss,
                "classification": candidate.score.classification,
                "vetoed": candidate.score.vetoed,
            }
            for candidate in scored_candidates
        ],
    }


def _parse_legal_move(board: chess.Board, raw_response: str) -> chess.Move | None:
    parsed = parse_uci_json(raw_response)
    if not parsed.parse_ok or parsed.move is None:
        return None
    try:
        move = chess.Move.from_uci(parsed.move.strip().lower())
    except ValueError:
        return None
    if move not in board.legal_moves:
        return None
    return move


def _safe_selection_key(candidate: ScoredCandidate) -> tuple[int, int, int]:
    return (
        -candidate.candidate.multiplicity,
        _cpl_sort_value(candidate),
        candidate.candidate.first_index,
    )


def _least_bad_selection_key(candidate: ScoredCandidate) -> tuple[int, int]:
    return (
        _cpl_sort_value(candidate),
        candidate.candidate.first_index,
    )


def _cpl_sort_value(candidate: ScoredCandidate) -> int:
    if candidate.score.centipawn_loss is not None:
        return candidate.score.centipawn_loss
    if candidate.score.vetoed:
        return 1_000_000
    return 0


def _aggregate_proposal(
    *,
    raw_response: str,
    latency_ms: float,
    proposals: Sequence[MoveProposal],
    metadata: dict[str, Any],
) -> MoveProposal:
    return MoveProposal(
        raw_response=raw_response,
        latency_ms=latency_ms,
        prompt_tokens=_sum_optional(proposal.prompt_tokens for proposal in proposals),
        completion_tokens=_sum_optional(proposal.completion_tokens for proposal in proposals),
        total_tokens=_sum_optional(proposal.total_tokens for proposal in proposals),
        thinking=_join_optional(proposal.thinking for proposal in proposals),
        thinking_used=any(proposal.thinking_used for proposal in proposals),
        metadata=metadata,
    )


def _sum_optional(values: Iterable[int | None]) -> int | None:
    total = 0
    found = False
    for value in values:
        if value is None:
            continue
        found = True
        total += value
    return total if found else None


def _join_optional(values: Iterable[str | None]) -> str | None:
    parts = [value for value in values if value]
    return "\n\n".join(parts) if parts else None

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any, Literal, Protocol

import chess

from arena_core.llm.base import GenerationOptions
from arena_core.move_sources import MoveProposal
from arena_core.parser import parse_uci_json
from arena_core.utils import close_if_present

DELIBERATIVE_SOURCE_PREFIX = "deliberative:"


class DeliberationCandidateSource(Protocol):
    name: str
    source_type: str

    async def propose(
        self,
        *,
        prompt: str,
        board: chess.Board,
        options: GenerationOptions | None = None,
    ) -> MoveProposal:
        """Return one raw model response for an arbitrary prompt."""


@dataclass(frozen=True)
class DeliberationConfig:
    mode: Literal["native_think", "revise", "candidate_critic", "candidate_pairwise"] = "revise"
    n_candidates: int = 5
    candidate_temperature: float = 0.7
    critic_temperature: float = 0.0
    final_temperature: float = 0.0
    max_opponent_replies: int = 40
    include_ascii_board: bool = True
    include_legal_moves: bool = True
    max_analysis_tokens: int = 1024
    max_final_tokens: int = 64
    max_pairwise_tokens: int = 24
    persist_intermediate_prompts: bool = True


@dataclass(frozen=True)
class DeliberationCandidate:
    move: chess.Move
    uci: str
    san: str
    idea: str | None
    raw: dict[str, Any]
    first_index: int


@dataclass(frozen=True)
class PairwiseDecision:
    left_uci: str
    right_uci: str
    selected_uci: str
    parsed_move: str | None
    parse_ok: bool
    candidate_ok: bool
    raw_response: str
    cache_hit: bool
    latency_ms: float
    fallback_reason: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "left": self.left_uci,
            "right": self.right_uci,
            "selected": self.selected_uci,
            "parsed_move": self.parsed_move,
            "parse_ok": self.parse_ok,
            "candidate_ok": self.candidate_ok,
            "raw_response": self.raw_response,
            "cache_hit": self.cache_hit,
            "latency_ms": self.latency_ms,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class PairwiseSelectionResult:
    selected: DeliberationCandidate
    vote_counts: dict[str, int]
    decisions: list[PairwiseDecision]
    prompts: list[str]
    proposals: list[MoveProposal]
    cache_hits: int
    invalid_count: int


class DeliberativeLLMMoveSource:
    source_type = "llm_deliberative"

    def __init__(
        self,
        *,
        inner: DeliberationCandidateSource,
        pairwise_critic: DeliberationCandidateSource | None = None,
        config: DeliberationConfig | None = None,
    ) -> None:
        self.inner = inner
        self.pairwise_critic = pairwise_critic
        self.config = config or DeliberationConfig()
        self.name = f"{DELIBERATIVE_SOURCE_PREFIX}{inner.name}"
        self._pairwise_cache: dict[tuple[str, str, str], PairwiseDecision] = {}

    async def propose(self, *, prompt: str, board: chess.Board) -> MoveProposal:
        if self.config.mode == "native_think":
            proposal = await self.inner.propose(prompt=prompt, board=board.copy())
            return _with_metadata(
                proposal,
                {
                    "regime": "llm_deliberative",
                    "mode": "native_think",
                    "inner_source": self.inner.name,
                },
            )
        if self.config.mode == "candidate_critic":
            return await self._candidate_critic(prompt=prompt, board=board)
        if self.config.mode == "candidate_pairwise":
            return await self._candidate_pairwise(prompt=prompt, board=board)
        return await self._revise(prompt=prompt, board=board)

    async def _revise(self, *, prompt: str, board: chess.Board) -> MoveProposal:
        started = time.perf_counter()
        initial = await self.inner.propose(prompt=prompt, board=board.copy())
        initial_move = _legal_move_from_response(board, initial.raw_response)
        metadata: dict[str, Any] = {
            "regime": "llm_deliberative",
            "mode": "revise",
            "inner_source": self.inner.name,
            "initial_raw_response": initial.raw_response,
            "initial_move": initial_move.uci() if initial_move is not None else None,
            "stage_token_usage": {"initial": _token_total(initial)},
            "stage_latency_ms": {"initial": initial.latency_ms},
        }
        if self.config.persist_intermediate_prompts:
            metadata["initial_prompt"] = prompt
        if initial_move is None:
            metadata["final_move"] = None
            metadata["changed_move"] = False
            return _aggregate_proposal(
                raw_response=initial.raw_response,
                started=started,
                proposals=[initial],
                metadata=metadata,
            )

        revise_prompt = _build_revise_prompt(
            board=board,
            candidate=initial_move,
            max_opponent_replies=self.config.max_opponent_replies,
        )
        revised = await self.inner.propose(
            prompt=revise_prompt,
            board=board.copy(),
            options=GenerationOptions(
                temperature=self.config.critic_temperature,
                num_predict=self.config.max_analysis_tokens,
            ),
        )
        final_move = _legal_move_from_response(board, revised.raw_response)
        revise_invalid_fallback_initial = final_move is None
        if final_move is None:
            final_move = initial_move
        final_uci = final_move.uci() if final_move is not None else None
        metadata.update(
            {
                "revise_raw_response": revised.raw_response,
                "final_move": final_uci,
                "changed_move": final_uci is not None and final_uci != initial_move.uci(),
                "revise_invalid_fallback_initial": revise_invalid_fallback_initial,
                "stage_token_usage": {
                    "initial": _token_total(initial),
                    "revise": _token_total(revised),
                },
                "stage_latency_ms": {
                    "initial": initial.latency_ms,
                    "revise": revised.latency_ms,
                },
            }
        )
        _copy_json_fields(
            revised.raw_response,
            metadata,
            {
                "changed": "self_reported_changed",
                "risk": "self_reported_risk",
            },
        )
        if self.config.persist_intermediate_prompts:
            metadata["revise_prompt"] = revise_prompt
        raw_response = json.dumps({"move": final_move.uci()}, separators=(",", ":"))
        return _aggregate_proposal(
            raw_response=raw_response,
            started=started,
            proposals=[initial, revised],
            metadata=metadata,
        )

    async def _candidate_critic(self, *, prompt: str, board: chess.Board) -> MoveProposal:
        started = time.perf_counter()
        candidate_prompt = _build_candidate_prompt(
            board=board,
            n_candidates=self.config.n_candidates,
            include_ascii_board=self.config.include_ascii_board,
            include_legal_moves=self.config.include_legal_moves,
        )
        candidate_proposal = await self.inner.propose(
            prompt=candidate_prompt,
            board=board.copy(),
            options=GenerationOptions(
                temperature=self.config.candidate_temperature,
                num_predict=self.config.max_analysis_tokens,
            ),
        )
        candidates = collect_legal_candidates_from_response(board, candidate_proposal.raw_response)
        metadata: dict[str, Any] = {
            "regime": "llm_deliberative",
            "mode": "candidate_critic",
            "inner_source": self.inner.name,
            "n_candidates_configured": self.config.n_candidates,
            "candidate_raw_response": candidate_proposal.raw_response,
            "n_distinct_legal_candidates": len(candidates),
            "candidate_legal_moves": [candidate.uci for candidate in candidates],
            "stage_token_usage": {"candidate": _token_total(candidate_proposal)},
            "stage_latency_ms": {"candidate": candidate_proposal.latency_ms},
        }
        if self.config.persist_intermediate_prompts:
            metadata["candidate_prompt"] = candidate_prompt
        if not candidates:
            fallback = await self.inner.propose(
                prompt=prompt,
                board=board.copy(),
                options=GenerationOptions(
                    temperature=self.config.final_temperature,
                    num_predict=self.config.max_final_tokens,
                ),
            )
            metadata["final_move"] = None
            metadata["changed_move"] = False
            metadata["no_legal_candidates_fallback_single_shot"] = True
            metadata["fallback_raw_response"] = fallback.raw_response
            metadata["stage_token_usage"] = {
                "candidate": _token_total(candidate_proposal),
                "fallback": _token_total(fallback),
            }
            metadata["stage_latency_ms"] = {
                "candidate": candidate_proposal.latency_ms,
                "fallback": fallback.latency_ms,
            }
            return _aggregate_proposal(
                raw_response=fallback.raw_response,
                started=started,
                proposals=[candidate_proposal, fallback],
                metadata=metadata,
            )

        critic_rows: list[dict[str, Any]] = []
        critic_proposals: list[MoveProposal] = []
        critic_prompts: list[str] = []
        for candidate in candidates:
            critic_prompt = _build_critic_prompt(
                board=board,
                candidate=candidate,
                max_opponent_replies=self.config.max_opponent_replies,
            )
            critic = await self.inner.propose(
                prompt=critic_prompt,
                board=board.copy(),
                options=GenerationOptions(
                    temperature=self.config.critic_temperature,
                    num_predict=self.config.max_analysis_tokens,
                ),
            )
            critic_proposals.append(critic)
            critic_prompts.append(critic_prompt)
            critic_rows.append(
                _critic_summary(board=board, candidate=candidate, raw=critic.raw_response)
            )

        final_prompt = _build_final_prompt(
            board=board,
            candidates=candidates,
            critic_summaries=critic_rows,
        )
        final_proposal = await self.inner.propose(
            prompt=final_prompt,
            board=board.copy(),
            options=GenerationOptions(
                temperature=self.config.final_temperature,
                num_predict=self.config.max_final_tokens,
            ),
        )
        final_move = _legal_candidate_from_response(
            board=board,
            raw_response=final_proposal.raw_response,
            candidates=candidates,
        )
        final_invalid_fallback_first_candidate = final_move is None
        if final_move is None:
            final_move = candidates[0].move
        final_uci = final_move.uci() if final_move is not None else None
        first_uci = candidates[0].uci if candidates else None
        metadata.update(
            {
                "critic_summaries": critic_rows,
                "final_raw_response": final_proposal.raw_response,
                "final_move": final_uci,
                "final_invalid_fallback_first_candidate": final_invalid_fallback_first_candidate,
                "changed_move": (
                    final_uci is not None and first_uci is not None and final_uci != first_uci
                ),
                "stage_token_usage": {
                    "candidate": _token_total(candidate_proposal),
                    "critics": sum(_token_total(proposal) for proposal in critic_proposals),
                    "final": _token_total(final_proposal),
                },
                "stage_latency_ms": {
                    "candidate": candidate_proposal.latency_ms,
                    "critics": sum(proposal.latency_ms for proposal in critic_proposals),
                    "final": final_proposal.latency_ms,
                },
            }
        )
        if self.config.persist_intermediate_prompts:
            metadata["critic_prompts"] = critic_prompts
            metadata["final_prompt"] = final_prompt
        proposals = [candidate_proposal, *critic_proposals, final_proposal]
        raw_response = (
            json.dumps({"move": final_move.uci()}, separators=(",", ":"))
            if final_move is not None
            else final_proposal.raw_response
        )
        return _aggregate_proposal(
            raw_response=raw_response,
            started=started,
            proposals=proposals,
            metadata=metadata,
        )

    async def _candidate_pairwise(self, *, prompt: str, board: chess.Board) -> MoveProposal:
        started = time.perf_counter()
        candidate_prompt = _build_candidate_prompt(
            board=board,
            n_candidates=self.config.n_candidates,
            include_ascii_board=self.config.include_ascii_board,
            include_legal_moves=self.config.include_legal_moves,
        )
        candidate_proposal = await self.inner.propose(
            prompt=candidate_prompt,
            board=board.copy(),
            options=GenerationOptions(
                temperature=self.config.candidate_temperature,
                num_predict=self.config.max_analysis_tokens,
            ),
        )
        candidates = collect_legal_candidates_from_response(board, candidate_proposal.raw_response)
        critic_source = self.pairwise_critic or self.inner
        metadata: dict[str, Any] = {
            "regime": "llm_deliberative",
            "mode": "candidate_pairwise",
            "inner_source": self.inner.name,
            "pairwise_critic_source": critic_source.name,
            "n_candidates_configured": self.config.n_candidates,
            "candidate_raw_response": candidate_proposal.raw_response,
            "n_distinct_legal_candidates": len(candidates),
            "candidate_legal_moves": [candidate.uci for candidate in candidates],
            "stage_token_usage": {"candidate": _token_total(candidate_proposal)},
            "stage_latency_ms": {"candidate": candidate_proposal.latency_ms},
        }
        if self.config.persist_intermediate_prompts:
            metadata["candidate_prompt"] = candidate_prompt
        if not candidates:
            fallback = await self.inner.propose(
                prompt=prompt,
                board=board.copy(),
                options=GenerationOptions(
                    temperature=self.config.final_temperature,
                    num_predict=self.config.max_final_tokens,
                ),
            )
            metadata["final_move"] = None
            metadata["changed_move"] = False
            metadata["no_legal_candidates_fallback_single_shot"] = True
            metadata["fallback_raw_response"] = fallback.raw_response
            metadata["stage_token_usage"] = {
                "candidate": _token_total(candidate_proposal),
                "fallback": _token_total(fallback),
            }
            metadata["stage_latency_ms"] = {
                "candidate": candidate_proposal.latency_ms,
                "fallback": fallback.latency_ms,
            }
            return _aggregate_proposal(
                raw_response=fallback.raw_response,
                started=started,
                proposals=[candidate_proposal, fallback],
                metadata=metadata,
            )

        pairwise_result = await self._run_pairwise_selector(
            board=board,
            candidates=candidates,
            critic_source=critic_source,
        )
        final_move = pairwise_result.selected.move
        final_uci = final_move.uci()
        first_uci = candidates[0].uci
        metadata.update(
            {
                "final_move": final_uci,
                "changed_move": final_uci != first_uci,
                "pairwise_vote_counts": pairwise_result.vote_counts,
                "pairwise_decisions": [
                    decision.to_metadata() for decision in pairwise_result.decisions
                ],
                "pairwise_prompt_count": len(pairwise_result.prompts),
                "pairwise_cache_hits": pairwise_result.cache_hits,
                "pairwise_invalid_count": pairwise_result.invalid_count,
                "selection_method": "pairwise_votes_tiebreak_generator_order",
                "stage_token_usage": {
                    "candidate": _token_total(candidate_proposal),
                    "pairwise": sum(
                        _token_total(proposal) for proposal in pairwise_result.proposals
                    ),
                },
                "stage_latency_ms": {
                    "candidate": candidate_proposal.latency_ms,
                    "pairwise": sum(proposal.latency_ms for proposal in pairwise_result.proposals),
                },
            }
        )
        if self.config.persist_intermediate_prompts:
            metadata["pairwise_prompts"] = pairwise_result.prompts
        raw_response = json.dumps({"move": final_uci}, separators=(",", ":"))
        return _aggregate_proposal(
            raw_response=raw_response,
            started=started,
            proposals=[candidate_proposal, *pairwise_result.proposals],
            metadata=metadata,
        )

    async def _run_pairwise_selector(
        self,
        *,
        board: chess.Board,
        candidates: Sequence[DeliberationCandidate],
        critic_source: DeliberationCandidateSource,
    ) -> "PairwiseSelectionResult":
        if len(candidates) == 1:
            return PairwiseSelectionResult(
                selected=candidates[0],
                vote_counts={candidates[0].uci: 0},
                decisions=[],
                prompts=[],
                proposals=[],
                cache_hits=0,
                invalid_count=0,
            )

        votes = {candidate.uci: 0 for candidate in candidates}
        by_uci = {candidate.uci: candidate for candidate in candidates}
        prompts: list[str] = []
        proposals: list[MoveProposal] = []
        decisions: list[PairwiseDecision] = []
        cache_hits = 0
        invalid_count = 0
        for left, right in combinations(candidates, 2):
            cache_key = (board.fen(), left.uci, right.uci)
            cached = self._pairwise_cache.get(cache_key)
            if cached is not None:
                decision = replace(cached, cache_hit=True, latency_ms=0.0)
                cache_hits += 1
            else:
                pairwise_prompt = _build_pairwise_prompt(
                    board=board,
                    candidates=[left, right],
                )
                prompts.append(pairwise_prompt)
                proposal = await critic_source.propose(
                    prompt=pairwise_prompt,
                    board=board.copy(),
                    options=GenerationOptions(
                        temperature=self.config.critic_temperature,
                        num_predict=self.config.max_pairwise_tokens,
                    ),
                )
                proposals.append(proposal)
                decision = _pairwise_decision_from_response(
                    board=board,
                    left=left,
                    right=right,
                    raw_response=proposal.raw_response,
                    cache_hit=False,
                    latency_ms=proposal.latency_ms,
                )
                self._pairwise_cache[cache_key] = decision
            if not decision.candidate_ok:
                invalid_count += 1
            votes[decision.selected_uci] += 1
            decisions.append(decision)

        selected_uci = max(
            votes,
            key=lambda uci: (votes[uci], -by_uci[uci].first_index),
        )
        return PairwiseSelectionResult(
            selected=by_uci[selected_uci],
            vote_counts=votes,
            decisions=decisions,
            prompts=prompts,
            proposals=proposals,
            cache_hits=cache_hits,
            invalid_count=invalid_count,
        )

    def close(self) -> None:
        close_if_present(self.inner)
        if self.pairwise_critic is not None and self.pairwise_critic is not self.inner:
            close_if_present(self.pairwise_critic)


def is_deliberative_source_name(name: str) -> bool:
    return name.startswith(DELIBERATIVE_SOURCE_PREFIX) and bool(
        name.removeprefix(DELIBERATIVE_SOURCE_PREFIX)
    )


def deliberative_inner_source_name(name: str) -> str:
    if is_deliberative_source_name(name):
        return name.removeprefix(DELIBERATIVE_SOURCE_PREFIX)
    return name


def collect_legal_candidates_from_response(
    board: chess.Board,
    raw_response: str,
) -> list[DeliberationCandidate]:
    payload = _json_object(raw_response)
    if payload is None:
        return []
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        return []
    by_uci: dict[str, DeliberationCandidate] = {}
    for index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, dict):
            continue
        move_text = raw_candidate.get("move")
        if not isinstance(move_text, str):
            continue
        move = _legal_move_from_text(board, move_text)
        if move is None:
            continue
        uci = move.uci()
        if uci in by_uci:
            continue
        by_uci[uci] = DeliberationCandidate(
            move=move,
            uci=uci,
            san=board.san(move),
            idea=raw_candidate.get("idea") if isinstance(raw_candidate.get("idea"), str) else None,
            raw=raw_candidate,
            first_index=index,
        )
    return sorted(by_uci.values(), key=lambda candidate: candidate.first_index)


def _build_candidate_prompt(
    *,
    board: chess.Board,
    n_candidates: int,
    include_ascii_board: bool,
    include_legal_moves: bool,
) -> str:
    sections = [
        "You are choosing candidate chess moves for the current position.",
        "Your task is NOT to choose a final move yet.",
        "Return a JSON object with a candidates array only.",
        "Do not return a top-level move field.",
        "Required shape:",
        '{"candidates":[{"move":"e2e4","idea":"claim center"}]}',
        f"Generate up to {n_candidates} distinct legal candidate moves.",
        "Do not use an engine. Use only the board, legal moves, and chess reasoning.",
        "",
        f"FEN: {board.fen()}",
        f"Side to move: {_side_name(board.turn)}",
        f"SAN history: {_san_history_text(board)}",
    ]
    if include_ascii_board:
        sections.extend(["Board:", str(board)])
    if include_legal_moves:
        sections.append(f"Legal moves: {_legal_moves_text(board)}")
    return "\n".join(sections)


def _build_revise_prompt(
    *,
    board: chess.Board,
    candidate: chess.Move,
    max_opponent_replies: int,
) -> str:
    after = board.copy(stack=False)
    candidate_san = board.san(candidate)
    after.push(candidate)
    return "\n".join(
        [
            "You are reviewing your own chess move before it is played.",
            "",
            f"Original position FEN: {board.fen()}",
            f"Side to move: {_side_name(board.turn)}",
            f"SAN history: {_san_history_text(board)}",
            f"Legal moves from original position: {_legal_moves_text(board)}",
            f"Your first chosen move: {candidate.uci()} ({candidate_san})",
            f"Position after your move: {after.fen()}",
            "Opponent legal replies after your move: "
            f"{_legal_moves_text(after, max_opponent_replies)}",
            "",
            "Check whether your first move is a blunder.",
            "Focus on forcing replies: checks, captures, threats, mate threats, "
            "attacks on the queen or king, hanging pieces, and ignored opponent threats.",
            "You may keep the move or replace it with a different legal move from "
            "the original legal move list.",
            "Return strict JSON only:",
            '{"move":"e2e4","changed":false,"risk":"short tactical risk note"}',
        ]
    )


def _build_critic_prompt(
    *,
    board: chess.Board,
    candidate: DeliberationCandidate,
    max_opponent_replies: int,
) -> str:
    after = board.copy(stack=False)
    after.push(candidate.move)
    return "\n".join(
        [
            "You are the opponent trying to refute a candidate chess move.",
            "",
            f"Original position FEN: {board.fen()}",
            f"Side to move before candidate: {_side_name(board.turn)}",
            f"Candidate move: {candidate.uci} ({candidate.san})",
            f"Candidate idea: {candidate.idea or 'none'}",
            f"Position after candidate: {after.fen()}",
            f"Opponent side to move: {_side_name(after.turn)}",
            f"Opponent legal replies: {_legal_moves_text(after, max_opponent_replies)}",
            "",
            "Find the most dangerous opponent reply. Look first for checks, captures, "
            "mate threats, attacks on queen or king, and tactics that win material.",
            "Do not use an engine. Reason from the board and legal moves only.",
            "Return strict JSON only:",
            '{"candidate":"'
            + candidate.uci
            + '","best_reply":"<uci or null>","risk":"low|medium|high|unknown",'
            '"blunder_suspected":true,"reason":"short concrete reason"}',
        ]
    )


def _build_final_prompt(
    *,
    board: chess.Board,
    candidates: Sequence[DeliberationCandidate],
    critic_summaries: Sequence[dict[str, Any]],
) -> str:
    candidate_table = "\n".join(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) for summary in critic_summaries
    )
    candidate_moves = ", ".join(candidate.uci for candidate in candidates)
    return "\n".join(
        [
            "You must now choose one final chess move.",
            "",
            f"Original position FEN: {board.fen()}",
            f"Side to move: {_side_name(board.turn)}",
            f"Legal moves: {_legal_moves_text(board)}",
            f"Candidate moves, exact UCI only: {candidate_moves}",
            "",
            "Candidate analyses:",
            candidate_table,
            "",
            "Choose the candidate with the lowest tactical risk.",
            "Avoid any move where your own critic found a forcing refutation, "
            "mate threat, or clear material loss.",
            "Your final move must be copied exactly from Candidate moves.",
            "If there is only one candidate, choose that exact UCI move.",
            "Do not return SAN, algebraic notation, or a move outside the candidate list.",
            "Return strict JSON only:",
            '{"move":"e2e4"}',
        ]
    )


def _build_pairwise_prompt(
    *,
    board: chess.Board,
    candidates: Sequence[DeliberationCandidate],
) -> str:
    candidate_lines = [
        f"{index}. {candidate.uci} ({candidate.san})"
        for index, candidate in enumerate(candidates, start=1)
    ]
    return "\n".join(
        [
            "You are a chess move safety comparator.",
            "",
            "Position:",
            f"FEN: {board.fen()}",
            f"Side to move: {_side_name(board.turn)}",
            "",
            "Candidate moves:",
            *candidate_lines,
            "",
            "Choose the safer move from the two candidates.",
            "Return JSON only:",
            '{"move":"<uci>"}',
        ]
    )


def _pairwise_decision_from_response(
    *,
    board: chess.Board,
    left: DeliberationCandidate,
    right: DeliberationCandidate,
    raw_response: str,
    cache_hit: bool,
    latency_ms: float,
) -> PairwiseDecision:
    parsed = parse_uci_json(raw_response)
    pair_moves = {left.uci, right.uci}
    selected_uci = left.uci
    candidate_ok = False
    fallback_reason: str | None = None
    if parsed.parse_ok and parsed.move is not None:
        move = _legal_move_from_text(board, parsed.move)
        if move is not None and move.uci() in pair_moves:
            selected_uci = move.uci()
            candidate_ok = True
        else:
            fallback_reason = "parsed_move_not_pair_candidate"
    else:
        fallback_reason = parsed.error_type or "parse_failed"
    return PairwiseDecision(
        left_uci=left.uci,
        right_uci=right.uci,
        selected_uci=selected_uci,
        parsed_move=parsed.move,
        parse_ok=parsed.parse_ok,
        candidate_ok=candidate_ok,
        raw_response=raw_response,
        cache_hit=cache_hit,
        latency_ms=latency_ms,
        fallback_reason=fallback_reason,
    )


def _critic_summary(
    *,
    board: chess.Board,
    candidate: DeliberationCandidate,
    raw: str,
) -> dict[str, Any]:
    payload = _json_object(raw) or {}
    after = board.copy(stack=False)
    after.push(candidate.move)
    best_reply = payload.get("best_reply")
    if isinstance(best_reply, str):
        best_reply_move = _legal_move_from_text(after, best_reply)
        best_reply_value: str | None = (
            best_reply_move.uci() if best_reply_move is not None else None
        )
    else:
        best_reply_value = None
    return {
        "candidate": candidate.uci,
        "san": candidate.san,
        "idea": candidate.idea,
        "best_reply": best_reply_value,
        "risk": payload.get("risk") if isinstance(payload.get("risk"), str) else "unknown",
        "blunder_suspected": (
            payload.get("blunder_suspected")
            if isinstance(payload.get("blunder_suspected"), bool)
            else None
        ),
        "reason": payload.get("reason") if isinstance(payload.get("reason"), str) else None,
        "raw_response": raw,
    }


def _aggregate_proposal(
    *,
    raw_response: str,
    started: float,
    proposals: Sequence[MoveProposal],
    metadata: dict[str, Any],
) -> MoveProposal:
    return MoveProposal(
        raw_response=raw_response,
        latency_ms=(time.perf_counter() - started) * 1000,
        prompt_tokens=_sum_optional(proposal.prompt_tokens for proposal in proposals),
        completion_tokens=_sum_optional(proposal.completion_tokens for proposal in proposals),
        total_tokens=_sum_optional(proposal.total_tokens for proposal in proposals),
        thinking="\n\n".join(
            proposal.thinking for proposal in proposals if proposal.thinking is not None
        )
        or None,
        thinking_used=any(proposal.thinking_used for proposal in proposals),
        metadata=metadata,
    )


def _with_metadata(proposal: MoveProposal, metadata: dict[str, Any]) -> MoveProposal:
    merged = {**metadata, **(proposal.metadata or {})}
    return MoveProposal(
        raw_response=proposal.raw_response,
        latency_ms=proposal.latency_ms,
        prompt_tokens=proposal.prompt_tokens,
        completion_tokens=proposal.completion_tokens,
        total_tokens=proposal.total_tokens,
        thinking=proposal.thinking,
        thinking_used=proposal.thinking_used,
        metadata=merged,
    )


def _legal_move_from_response(board: chess.Board, raw_response: str) -> chess.Move | None:
    parsed = parse_uci_json(raw_response)
    if not parsed.parse_ok or parsed.move is None:
        return None
    return _legal_move_from_text(board, parsed.move)


def _legal_candidate_from_response(
    *,
    board: chess.Board,
    raw_response: str,
    candidates: Sequence[DeliberationCandidate],
) -> chess.Move | None:
    parsed = parse_uci_json(raw_response)
    if not parsed.parse_ok or parsed.move is None:
        return None
    candidate_uci = {candidate.uci for candidate in candidates}
    move = _legal_move_from_text(board, parsed.move)
    if move is not None and move.uci() in candidate_uci:
        return move
    try:
        san_move = board.parse_san(parsed.move.strip())
    except ValueError:
        return None
    if san_move.uci() in candidate_uci:
        return san_move
    return None


def _legal_move_from_text(board: chess.Board, text: str) -> chess.Move | None:
    try:
        move = chess.Move.from_uci(text.strip().lower())
    except ValueError:
        return None
    return move if move in board.legal_moves else None


def _json_object(raw_response: str) -> dict[str, Any] | None:
    normalized = raw_response.strip()
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        object_text = _first_balanced_object(normalized)
        if object_text is None:
            return None
        try:
            payload = json.loads(object_text)
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def _first_balanced_object(value: str) -> str | None:
    start = value.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(value[start:], start=start):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return value[start : index + 1]
    return None


def _copy_json_fields(
    raw_response: str,
    metadata: dict[str, Any],
    field_map: dict[str, str],
) -> None:
    payload = _json_object(raw_response)
    if payload is None:
        return
    for source, target in field_map.items():
        if source in payload:
            metadata[target] = payload[source]


def _legal_moves_text(board: chess.Board, limit: int | None = None) -> str:
    moves = sorted(move.uci() for move in board.legal_moves)
    if limit is not None and limit >= 0 and len(moves) > limit:
        visible = moves[:limit]
        return f"{', '.join(visible)} ... ({len(moves) - limit} more)"
    return ", ".join(moves)


def _side_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


def _san_history_text(board: chess.Board) -> str:
    if not board.move_stack:
        return "(none)"
    replay = chess.Board()
    san_moves: list[str] = []
    for move in board.move_stack:
        if move not in replay.legal_moves:
            return " ".join(stack_move.uci() for stack_move in board.move_stack)
        san_moves.append(replay.san(move))
        replay.push(move)
    return " ".join(san_moves)


def _sum_optional(values: Sequence[int | None] | Any) -> int | None:
    concrete = [value for value in values if value is not None]
    return sum(concrete) if concrete else None


def _token_total(proposal: MoveProposal) -> int:
    return proposal.total_tokens or 0

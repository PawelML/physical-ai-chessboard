import json
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from inspect import isawaitable
from typing import Protocol

import chess
import chess.pgn
from sqlalchemy.ext.asyncio import AsyncSession

from arena_core.config import Settings
from arena_core.evaluators.stockfish import EngineEvaluation
from arena_core.llm.base import LLMService
from arena_core.move_sources import MoveProposal
from arena_core.parser import ParsedMove, parse_uci_json
from arena_core.persistence import models
from arena_core.persistence.repositories import ensure_prompt, text_hash
from arena_core.prompts import LegalityMode, build_strict_prompt
from arena_core.telemetry import estimate_usage


class MoveSource(Protocol):
    name: str
    source_type: str

    async def propose(self, *, prompt: str, board: chess.Board) -> MoveProposal:
        """Return one raw move proposal."""


class MoveEvaluator(Protocol):
    def evaluate_move(self, board_before: chess.Board, move: chess.Move) -> EngineEvaluation:
        """Evaluate one accepted move from the moving side's point of view."""


class RandomMoveSource:
    name = "random"
    source_type = "random"

    def __init__(self, *, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    async def propose(self, *, prompt: str, board: chess.Board) -> MoveProposal:
        started = time.perf_counter()
        move = self._rng.choice(list(board.legal_moves)).uci()
        latency_ms = (time.perf_counter() - started) * 1000
        return MoveProposal(
            raw_response=f'{{"move":"{move}"}}',
            latency_ms=latency_ms,
        )


class StaticMoveSource:
    source_type = "llm"

    def __init__(self, responses: list[str], *, name: str = "static") -> None:
        self.responses = responses
        self.name = name
        self._idx = 0

    async def propose(self, *, prompt: str, board: chess.Board) -> MoveProposal:
        started = time.perf_counter()
        if self._idx >= len(self.responses):
            response = '{"move":"0000"}'
        else:
            response = self.responses[self._idx]
            self._idx += 1
        return MoveProposal(
            raw_response=response,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


class _HumanMoveSource:
    name = "human"
    source_type = "human"

    async def propose(self, *, prompt: str, board: chess.Board) -> MoveProposal:
        raise RuntimeError("human moves are submitted directly")


class LLMMoveSource:
    source_type = "llm"

    def __init__(self, *, model: str, service: LLMService) -> None:
        self.name = model
        self._model = model
        self._service = service

    async def propose(self, *, prompt: str, board: chess.Board) -> MoveProposal:
        started = time.perf_counter()
        response = await self._service.complete(model=self._model, prompt=prompt)
        latency_ms = (time.perf_counter() - started) * 1000
        return MoveProposal(
            raw_response=response.content,
            latency_ms=latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            thinking=response.thinking,
            thinking_used=response.thinking_used,
        )


@dataclass
class GameResult:
    game_id: int
    result: str
    termination_reason: str
    pgn: str
    plies: int


class ArenaGame:
    def __init__(
        self,
        *,
        white: MoveSource,
        black: MoveSource,
        settings: Settings,
        legality_mode: LegalityMode = "open",
        max_plies: int | None = None,
        evaluator: MoveEvaluator | None = None,
        initial_board: chess.Board | None = None,
        run_id: int | None = None,
        white_participant_id: int | None = None,
        black_participant_id: int | None = None,
        opening_line_id: int | None = None,
        strategic_memory: bool = False,
    ) -> None:
        self.white = white
        self.black = black
        self.settings = settings
        self.legality_mode: LegalityMode = legality_mode
        self.max_plies = max_plies
        self.evaluator = evaluator
        self.board = initial_board.copy(stack=False) if initial_board is not None else chess.Board()
        self.run_id = run_id
        self.white_participant_id = white_participant_id
        self.black_participant_id = black_participant_id
        self.opening_line_id = opening_line_id
        self.strategic_memory = strategic_memory
        self._initial_fen = self.board.fen()
        self._initial_ply = self.board.ply()
        self._san_history: list[str] = []
        self._uci_history: list[str] = []
        self._strategic_memory: dict[chess.Color, dict[str, str]] = {
            chess.WHITE: _initial_strategy_memory("white"),
            chess.BLACK: _initial_strategy_memory("black"),
        }

    async def run(
        self,
        session: AsyncSession,
        *,
        commit_after_each_ply: bool = False,
        on_game_started: Callable[[int], Awaitable[None] | None] | None = None,
    ) -> GameResult:
        game_row = await self.start(session)
        if commit_after_each_ply:
            await session.commit()
        if on_game_started is not None:
            callback_result = on_game_started(game_row.id)
            if isawaitable(callback_result):
                await callback_result

        termination_reason = "unknown"
        try:
            while not self.board.is_game_over(claim_draw=True):
                played_plies = self.board.ply() - self._initial_ply
                if self.max_plies is not None and played_plies >= self.max_plies:
                    termination_reason = "max_plies"
                    break
                move_source = self.white if self.board.turn == chess.WHITE else self.black
                accepted = await self.play_source_move(session, game_row.id, move_source)
                if accepted and commit_after_each_ply:
                    game_row.final_fen = self.board.fen()
                    game_row.pgn = self._export_pgn("*")
                if commit_after_each_ply:
                    await session.commit()
                if not accepted:
                    termination_reason = "forfeit_invalid"
                    break
            else:
                termination_reason = self._termination_reason()
        except Exception:
            if commit_after_each_ply:
                self.finish(game_row, termination_reason="error", result="*")
                await session.commit()
            raise
        finally:
            _close_if_present(self.white)
            if self.black is not self.white:
                _close_if_present(self.black)

        self.finish(game_row, termination_reason=termination_reason)
        if commit_after_each_ply:
            await session.commit()
        else:
            await session.flush()
        return GameResult(
            game_id=game_row.id,
            result=game_row.result,
            termination_reason=termination_reason,
            pgn=game_row.pgn or "",
            plies=self.board.ply() - self._initial_ply,
        )

    async def start(self, session: AsyncSession) -> models.Game:
        game_row = models.Game(
            run_id=self.run_id,
            white_participant_id=self.white_participant_id,
            black_participant_id=self.black_participant_id,
            opening_line_id=self.opening_line_id,
            final_fen=self.board.fen(),
            pgn=self._export_pgn("*"),
        )
        session.add(game_row)
        await session.flush()
        return game_row

    async def play_source_move(
        self,
        session: AsyncSession,
        game_id: int,
        move_source: MoveSource,
    ) -> bool:
        feedback: dict[str, object] | None = None
        attempt_rows: list[models.Attempt] = []
        latency_total_ms = 0.0
        max_attempts = self.settings.max_retries + 1
        for attempt_number in range(1, max_attempts + 1):
            prompt = build_strict_prompt(
                board=self.board,
                san_history=self._san_history,
                own_moves=self._own_moves_for_side(self.board.turn),
                last_opponent_move=self._last_opponent_move_for_side(self.board.turn),
                legality_mode=self.legality_mode,
                feedback=feedback,
                strategic_memory=(
                    self._strategic_memory[self.board.turn] if self.strategic_memory else None
                ),
                repetition_warning=(
                    self._repetition_warning() if self.strategic_memory else None
                ),
                version=self.settings.prompt_version,
            )
            prompt_row = await ensure_prompt(session, prompt)
            proposal = await move_source.propose(prompt=prompt.text, board=self.board.copy())
            latency_total_ms += proposal.latency_ms
            parsed = parse_uci_json(proposal.raw_response)
            legal_ok, legal_reason, chess_move = self._validate_move(parsed)
            feedback_for_attempt = feedback
            attempt_row = models.Attempt(
                game_id=game_id,
                ply=self.board.ply() + 1,
                attempt_number=attempt_number,
                prompt_id=prompt_row.id,
                raw_prompt_hash=text_hash(prompt.text),
                raw_prompt=prompt.text if self.settings.prompt_retention_enabled else None,
                raw_response=proposal.raw_response,
                parsed_move=parsed.move,
                parse_ok=parsed.parse_ok,
                legal_ok=legal_ok,
                error_type=None if legal_ok else parsed.error_type or "illegal_move",
                feedback_given=feedback_for_attempt,
                latency_ms=proposal.latency_ms,
                thinking=proposal.thinking,
                thinking_used=proposal.thinking_used,
            )
            session.add(attempt_row)
            await session.flush()
            self._add_token_usage(session, attempt_row.id, prompt.text, proposal)
            attempt_rows.append(attempt_row)
            if legal_ok and chess_move is not None:
                await self._accept_move(
                    session=session,
                    game_id=game_id,
                    move=chess_move,
                    move_source=move_source,
                    parsed=parsed,
                    retries_used=attempt_number - 1,
                    latency_total_ms=latency_total_ms,
                    attempts=attempt_rows,
                )
                return True
            remaining = max_attempts - attempt_number
            if remaining <= 0:
                return False
            feedback = {
                "error": parsed.error_type or "illegal_move",
                "attempted_move": parsed.move,
                "reason": parsed.reason or legal_reason,
                "legal_moves": sorted(move.uci() for move in self.board.legal_moves),
                "remaining_retries": remaining,
            }
        return False

    async def play_human_move(
        self,
        session: AsyncSession,
        game_id: int,
        move_text: str,
    ) -> tuple[bool, str | None]:
        started = time.perf_counter()
        raw_response = json.dumps({"move": move_text.strip()})
        parsed = parse_uci_json(raw_response)
        legal_ok, legal_reason, chess_move = self._validate_move(parsed)
        prompt_text = "Human move input"
        attempt_row = models.Attempt(
            game_id=game_id,
            ply=self.board.ply() + 1,
            attempt_number=1,
            prompt_id=None,
            raw_prompt_hash=text_hash(prompt_text),
            raw_prompt=prompt_text if self.settings.prompt_retention_enabled else None,
            raw_response=raw_response,
            parsed_move=parsed.move,
            parse_ok=parsed.parse_ok,
            legal_ok=legal_ok,
            error_type=None if legal_ok else parsed.error_type or "illegal_move",
            feedback_given=None,
            latency_ms=(time.perf_counter() - started) * 1000,
            thinking=None,
            thinking_used=False,
        )
        session.add(attempt_row)
        await session.flush()
        self._add_token_usage(
            session,
            attempt_row.id,
            prompt_text,
            MoveProposal(
                raw_response=raw_response,
                latency_ms=attempt_row.latency_ms,
            ),
        )
        if not legal_ok or chess_move is None:
            return False, parsed.reason or legal_reason
        await self._accept_move(
            session=session,
            game_id=game_id,
            move=chess_move,
            move_source=_HumanMoveSource(),
            parsed=parsed,
            retries_used=0,
            latency_total_ms=attempt_row.latency_ms,
            attempts=[attempt_row],
        )
        return True, None

    def pending_result(self) -> tuple[bool, str | None]:
        played_plies = self.board.ply() - self._initial_ply
        if self.max_plies is not None and played_plies >= self.max_plies:
            return True, "max_plies"
        if self.board.is_game_over(claim_draw=True):
            return True, self._termination_reason()
        return False, None

    def finish(
        self,
        game_row: models.Game,
        *,
        termination_reason: str,
        result: str | None = None,
    ) -> None:
        if result is None:
            if termination_reason == "max_plies":
                result = "*"
            elif termination_reason == "forfeit_invalid":
                result = "0-1" if self.board.turn == chess.WHITE else "1-0"
            else:
                result = self.board.result(claim_draw=True)
        game_row.result = result
        game_row.termination_reason = termination_reason
        game_row.final_fen = self.board.fen()
        game_row.pgn = self._export_pgn(game_row.result)
        game_row.ended_at = models.utcnow()

    def _add_token_usage(
        self,
        session: AsyncSession,
        attempt_id: int,
        prompt: str,
        proposal: MoveProposal,
    ) -> None:
        estimated = estimate_usage(
            prompt,
            proposal.raw_response,
            self.settings.default_context_window,
        )
        if proposal.prompt_tokens is not None:
            prompt_tokens = proposal.prompt_tokens
        else:
            prompt_tokens = estimated.prompt_tokens
        completion_tokens = (
            proposal.completion_tokens
            if proposal.completion_tokens is not None
            else estimated.completion_tokens
        )
        if proposal.total_tokens is not None:
            total_tokens = proposal.total_tokens
        else:
            total_tokens = prompt_tokens + completion_tokens
        session.add(
            models.TokenUsage(
                attempt_id=attempt_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_context_window=estimated.estimated_context_window,
                estimated_context_remaining=max(
                    estimated.estimated_context_window - total_tokens,
                    0,
                ),
                truncation_applied=estimated.truncation_applied,
                cost_usd=0.0,
            )
        )

    def _validate_move(self, parsed: ParsedMove) -> tuple[bool, str | None, chess.Move | None]:
        if not parsed.parse_ok or parsed.move is None:
            return False, parsed.reason, None
        move = self._parse_uci_move(parsed.move)
        if move is None:
            return False, f"{parsed.move} is not valid UCI syntax", None
        if move not in self.board.legal_moves:
            return False, f"{parsed.move} is not legal in the current position", None
        return True, None, move

    def _parse_uci_move(self, text: str) -> chess.Move | None:
        """Accept only raw UCI so models have a single move contract."""
        candidate = text.strip()
        try:
            return chess.Move.from_uci(candidate.lower())
        except ValueError:
            return None

    async def _accept_move(
        self,
        *,
        session: AsyncSession,
        game_id: int,
        move: chess.Move,
        move_source: MoveSource,
        parsed: ParsedMove,
        retries_used: int,
        latency_total_ms: float,
        attempts: list[models.Attempt],
    ) -> None:
        fen_before = self.board.fen()
        legal_move_count = self.board.legal_moves.count()
        san = self.board.san(move)
        color = "white" if self.board.turn == chess.WHITE else "black"
        self.board.push(move)
        fen_after = self.board.fen()
        move_row = models.Move(
            game_id=game_id,
            ply=self.board.ply(),
            color=color,
            fen_before=fen_before,
            fen_after=fen_after,
            accepted_uci=move.uci(),
            accepted_san=san,
            legal_move_count=legal_move_count,
            move_source=move_source.source_type,
            retries_used=retries_used,
            latency_total_ms=latency_total_ms,
        )
        session.add(move_row)
        await session.flush()
        if self.evaluator is not None:
            evaluation = self.evaluator.evaluate_move(chess.Board(fen_before), move)
            session.add(
                models.EngineEvaluation(
                    move_id=move_row.id,
                    engine_name=evaluation.engine_name,
                    engine_version=evaluation.engine_version,
                    nodes=evaluation.nodes,
                    depth_reached=evaluation.depth_reached,
                    eval_before_cp=evaluation.eval_before_cp,
                    eval_after_cp=evaluation.eval_after_cp,
                    mate_before=evaluation.mate_before,
                    mate_after=evaluation.mate_after,
                    best_move_uci=evaluation.best_move_uci,
                    centipawn_loss=evaluation.centipawn_loss,
                    classification=evaluation.classification,
                )
            )
        for attempt in attempts:
            if attempt.legal_ok:
                attempt.move_id = move_row.id
        self._san_history.append(san)
        self._uci_history.append(move.uci())
        if self.strategic_memory:
            self._update_strategic_memory(color, san, move.uci(), parsed)

    def _own_moves_for_side(self, side: chess.Color) -> list[tuple[str, str]]:
        start = 0 if side == chess.WHITE else 1
        return list(zip(self._san_history[start::2], self._uci_history[start::2], strict=True))

    def _last_opponent_move_for_side(self, side: chess.Color) -> str | None:
        if not self._san_history:
            return None
        last_was_white = len(self._san_history) % 2 == 1
        if side == chess.WHITE and not last_was_white:
            return f"{self._san_history[-1]}/{self._uci_history[-1]}"
        if side == chess.BLACK and last_was_white:
            return f"{self._san_history[-1]}/{self._uci_history[-1]}"
        return None

    def _repetition_warning(self) -> str | None:
        if self.board.can_claim_threefold_repetition():
            return (
                "A draw can be claimed by repetition. Avoid repeating unless a draw is your plan."
            )
        if self.board.is_repetition(2):
            return "The current position has repeated. Choose a move that improves the plan."
        return None

    def _update_strategic_memory(self, color: str, san: str, uci: str, parsed: ParsedMove) -> None:
        side = chess.WHITE if color == "white" else chess.BLACK
        memory = self._strategic_memory[side].copy()
        update = parsed.strategy_update or {}
        for key in ("objective", "opponent_threats", "pieces_to_improve", "avoid"):
            value = update.get(key)
            if isinstance(value, str) and value.strip():
                memory[key] = _compact_strategy_text(value)
        if parsed.rationale:
            memory["last_rationale"] = _compact_strategy_text(f"{san}/{uci}: {parsed.rationale}")
        else:
            memory["last_rationale"] = f"{san}/{uci}: no rationale returned"
        self._strategic_memory[side] = memory

    def _termination_reason(self) -> str:
        if self.board.is_checkmate():
            return "checkmate"
        if self.board.is_stalemate():
            return "stalemate"
        if self.board.is_insufficient_material():
            return "insufficient_material"
        if self.board.is_seventyfive_moves():
            return "seventyfive_moves"
        if self.board.is_fivefold_repetition():
            return "fivefold_repetition"
        if self.board.can_claim_draw():
            return "draw_claim"
        return "game_over"

    def _export_pgn(self, result: str) -> str:
        game = chess.pgn.Game()
        game.headers["Event"] = "Software Arena"
        game.headers["White"] = self.white.name
        game.headers["Black"] = self.black.name
        game.headers["Result"] = result
        if self._initial_fen != chess.STARTING_FEN:
            game.headers["SetUp"] = "1"
            game.headers["FEN"] = self._initial_fen
        node: chess.pgn.GameNode = game
        for uci in self._uci_history:
            move = chess.Move.from_uci(uci)
            node = node.add_variation(move)
        return str(game)


def _close_if_present(source: object) -> None:
    close = getattr(source, "close", None)
    if callable(close):
        close()


def _initial_strategy_memory(side: str) -> dict[str, str]:
    return {
        "objective": (
            f"Play active legal chess as {side}; develop pieces, keep the king safe, "
            "and improve the position."
        ),
        "opponent_threats": "Identify direct checks, captures, and attacks before choosing a move.",
        "pieces_to_improve": (
            "Develop undeveloped minor pieces and coordinate queen, rooks, and king safety."
        ),
        "avoid": (
            "Avoid moving the same piece repeatedly, exposing the king, "
            "or repeating positions without purpose."
        ),
        "last_rationale": "(none)",
    }


def _compact_strategy_text(value: str, *, limit: int = 220) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "..."

"""Sample legal candidate moves from a policy model for critic-ranker data."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess

from arena_core.deliberation import (
    _build_candidate_prompt,
    collect_legal_candidates_from_response,
)
from arena_core.llm.base import GenerationOptions, LLMService
from arena_core.llm.ollama import OllamaLLMService
from finetune._common import write_metadata_sidecar


@dataclass(frozen=True)
class PolicyCandidateSampleConfig:
    input: str | None
    output: str
    metadata_output: str | None
    model: str
    arena_db: str | None = None
    run_ids: list[int] = field(default_factory=list)
    split: str | None = "train"
    position_offset: int = 0
    max_positions: int | None = None
    samples_per_position: int = 1
    n_candidates: int = 5
    temperature: float = 0.7
    num_predict: int = 1024
    include_ascii_board: bool = True
    include_legal_moves: bool = True
    ollama_base_url: str = "http://localhost:11434"
    timeout_seconds: float = 120.0
    num_ctx: int | None = None
    num_gpu: int | None = None
    progress_every: int | None = None


@dataclass
class PolicyCandidateSampleStats:
    input_rows: int = 0
    arena_db_rows: int = 0
    positions_seen: int = 0
    positions_skipped_by_offset: int = 0
    positions_from_input: int = 0
    positions_from_arena_db: int = 0
    positions_sampled: int = 0
    requests: int = 0
    rows_written: int = 0
    legal_candidates_seen: int = 0
    dropped_duplicate_candidate: int = 0
    dropped_no_legal_candidates: int = 0
    service_errors: int = 0
    by_candidate_count: Counter[int] = field(default_factory=Counter)

    def to_json(self) -> dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "arena_db_rows": self.arena_db_rows,
            "positions_seen": self.positions_seen,
            "positions_skipped_by_offset": self.positions_skipped_by_offset,
            "positions_from_input": self.positions_from_input,
            "positions_from_arena_db": self.positions_from_arena_db,
            "positions_sampled": self.positions_sampled,
            "requests": self.requests,
            "rows_written": self.rows_written,
            "legal_candidates_seen": self.legal_candidates_seen,
            "dropped_duplicate_candidate": self.dropped_duplicate_candidate,
            "dropped_no_legal_candidates": self.dropped_no_legal_candidates,
            "service_errors": self.service_errors,
            "by_candidate_count": dict(sorted(self.by_candidate_count.items())),
        }


async def sample_policy_candidates(
    config: PolicyCandidateSampleConfig,
    *,
    service: LLMService,
) -> PolicyCandidateSampleStats:
    stats = PolicyCandidateSampleStats()
    positions = _read_positions(config, stats=stats)
    stats.positions_seen = len(positions)
    output_path = Path(config.output)
    output_rows: list[dict[str, Any]] = []
    for fen in positions:
        board = chess.Board(fen)
        seen_moves: set[str] = set()
        wrote_for_position = 0
        for sample_index in range(config.samples_per_position):
            stats.requests += 1
            prompt = _build_candidate_prompt(
                board=board,
                n_candidates=config.n_candidates,
                include_ascii_board=config.include_ascii_board,
                include_legal_moves=config.include_legal_moves,
            )
            try:
                response = await service.complete(
                    model=config.model,
                    prompt=prompt,
                    options=GenerationOptions(
                        temperature=config.temperature,
                        num_predict=config.num_predict,
                    ),
                )
            except Exception:
                stats.service_errors += 1
                continue
            candidates = collect_legal_candidates_from_response(board, response.content)
            stats.by_candidate_count[len(candidates)] += 1
            if not candidates:
                stats.dropped_no_legal_candidates += 1
                continue
            for candidate in candidates:
                stats.legal_candidates_seen += 1
                if candidate.uci in seen_moves:
                    stats.dropped_duplicate_candidate += 1
                    continue
                seen_moves.add(candidate.uci)
                output_rows.append(
                    _candidate_row(
                        board=board,
                        candidate=candidate,
                        model=config.model,
                        sample_index=sample_index,
                        raw_response=response.content,
                    )
                )
                stats.rows_written += 1
                wrote_for_position += 1
        if wrote_for_position > 0:
            stats.positions_sampled += 1
        if (
            config.progress_every is not None
            and config.progress_every > 0
            and stats.requests % config.progress_every == 0
        ):
            print(
                "sampled "
                f"{stats.requests} requests, {stats.rows_written} candidate rows, "
                f"{stats.service_errors} service errors",
                flush=True,
            )
    await asyncio.to_thread(_write_jsonl, output_path, output_rows)
    return stats


def _candidate_row(
    *,
    board: chess.Board,
    candidate: Any,
    model: str,
    sample_index: int,
    raw_response: str,
) -> dict[str, Any]:
    return {
        "fen_before": board.fen(),
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "candidate_uci": candidate.uci,
        "candidate_san": candidate.san,
        "source": "policy_sample",
        "candidate_rank_in_generator": candidate.first_index,
        "generator_idea": candidate.idea,
        "policy_model": model,
        "sample_index": sample_index,
        "candidate_raw": candidate.raw,
        "raw_response": raw_response,
    }


def _read_positions(
    config: PolicyCandidateSampleConfig,
    *,
    stats: PolicyCandidateSampleStats,
) -> list[str]:
    if config.input is None and config.arena_db is None:
        raise ValueError("Either input or arena_db must be configured.")

    positions: list[str] = []
    seen: set[str] = set()
    if config.input is not None:
        _append_jsonl_positions(
            Path(config.input),
            split=config.split,
            position_offset=config.position_offset,
            max_positions=config.max_positions,
            stats=stats,
            positions=positions,
            seen=seen,
        )
    if config.arena_db is not None:
        _append_arena_db_positions(
            Path(config.arena_db),
            run_ids=config.run_ids,
            position_offset=config.position_offset,
            max_positions=config.max_positions,
            stats=stats,
            positions=positions,
            seen=seen,
        )
    return positions


def _append_jsonl_positions(
    path: Path,
    *,
    split: str | None,
    position_offset: int,
    max_positions: int | None,
    stats: PolicyCandidateSampleStats,
    positions: list[str],
    seen: set[str],
) -> None:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if _position_limit_reached(positions, max_positions):
                break
            if not line.strip():
                continue
            stats.input_rows += 1
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            if split is not None and payload.get("split") != split:
                continue
            fen = payload.get("fen_before") or payload.get("fen")
            _maybe_append_position(
                fen,
                source="input",
                position_offset=position_offset,
                max_positions=max_positions,
                stats=stats,
                positions=positions,
                seen=seen,
            )


def _append_arena_db_positions(
    path: Path,
    *,
    run_ids: list[int],
    position_offset: int,
    max_positions: int | None,
    stats: PolicyCandidateSampleStats,
    positions: list[str],
    seen: set[str],
) -> None:
    where, params = _run_filter("g.run_id", run_ids)
    sql = f"""
        select
            m.fen_before
        from moves m
        join games g on g.id = m.game_id
        {where}
        order by g.run_id, g.id, m.ply
    """
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        for row in con.execute(sql, params):
            if _position_limit_reached(positions, max_positions):
                break
            stats.arena_db_rows += 1
            fen = row["fen_before"]
            _maybe_append_position(
                fen,
                source="arena_db",
                position_offset=position_offset,
                max_positions=max_positions,
                stats=stats,
                positions=positions,
                seen=seen,
            )
    finally:
        con.close()


def _maybe_append_position(
    fen: object,
    *,
    source: str,
    position_offset: int,
    max_positions: int | None,
    stats: PolicyCandidateSampleStats,
    positions: list[str],
    seen: set[str],
) -> None:
    if _position_limit_reached(positions, max_positions):
        return
    if not isinstance(fen, str) or fen in seen:
        return
    chess.Board(fen)
    seen.add(fen)
    if stats.positions_skipped_by_offset < position_offset:
        stats.positions_skipped_by_offset += 1
        return
    positions.append(fen)
    if source == "input":
        stats.positions_from_input += 1
    elif source == "arena_db":
        stats.positions_from_arena_db += 1
    else:
        raise ValueError(f"Unknown position source: {source}")


def _run_filter(column: str, run_ids: list[int]) -> tuple[str, list[int]]:
    if not run_ids:
        return "", []
    placeholders = ", ".join("?" for _ in run_ids)
    return f"where {column} in ({placeholders})", list(run_ids)


def _position_limit_reached(positions: list[str], max_positions: int | None) -> bool:
    return max_positions is not None and len(positions) >= max_positions


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, separators=(",", ":")) + "\n")


async def _amain() -> None:
    args = _parse_args()
    config = PolicyCandidateSampleConfig(
        input=str(args.input) if args.input else None,
        output=str(args.output),
        metadata_output=str(args.metadata_output) if args.metadata_output else None,
        model=args.model,
        arena_db=str(args.arena_db) if args.arena_db else None,
        run_ids=args.run_id,
        split=args.split,
        position_offset=args.position_offset,
        max_positions=args.max_positions,
        samples_per_position=args.samples_per_position,
        n_candidates=args.n_candidates,
        temperature=args.temperature,
        num_predict=args.num_predict,
        include_ascii_board=not args.no_ascii_board,
        include_legal_moves=not args.no_legal_moves,
        ollama_base_url=args.ollama_base_url,
        timeout_seconds=args.timeout_seconds,
        num_ctx=args.num_ctx,
        num_gpu=args.num_gpu,
        progress_every=args.progress_every,
    )
    service = OllamaLLMService(
        base_url=config.ollama_base_url,
        timeout_seconds=config.timeout_seconds,
        temperature=config.temperature,
        num_ctx=config.num_ctx,
        num_predict=config.num_predict,
        num_gpu=config.num_gpu,
        think="off",
    )
    stats = await sample_policy_candidates(config, service=service)
    if config.metadata_output is not None:
        write_metadata_sidecar(Path(config.metadata_output), config=config, stats=stats)
    print(f"Wrote {stats.rows_written} policy candidate rows to {config.output}.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample policy candidate moves from Ollama.")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arena-db", type=Path)
    parser.add_argument("--run-id", type=int, action="append", default=[])
    parser.add_argument("--split", default="train")
    parser.add_argument("--position-offset", type=int, default=0)
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--samples-per-position", type=int, default=1)
    parser.add_argument("--n-candidates", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--no-ascii-board", action="store_true")
    parser.add_argument("--no-legal-moves", action="store_true")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--num-ctx", type=int)
    parser.add_argument("--num-gpu", type=int)
    parser.add_argument("--progress-every", type=int)
    return parser.parse_args()


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()

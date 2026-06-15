import time
from dataclasses import dataclass

import chess
import chess.engine

from arena_core.move_sources import MoveProposal


@dataclass(frozen=True)
class EngineEvaluation:
    engine_name: str
    engine_version: str
    nodes: int
    depth_reached: int | None
    eval_before_cp: int | None
    eval_after_cp: int | None
    mate_before: int | None
    mate_after: int | None
    best_move_uci: str | None
    centipawn_loss: int | None
    classification: str


class StockfishEvaluator:
    def __init__(
        self,
        *,
        binary_path: str,
        nodes: int = 200_000,
        threads: int = 1,
        hash_mb: int = 128,
    ) -> None:
        self.binary_path = binary_path
        self.nodes = nodes
        self.threads = threads
        self.hash_mb = hash_mb
        self.engine_name = "stockfish"
        self._engine: chess.engine.SimpleEngine | None = None
        self._version: str | None = None

    def __enter__(self) -> "StockfishEvaluator":
        self._ensure_engine()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            finally:
                self._engine = None

    @property
    def version(self) -> str:
        engine = self._ensure_engine()
        if self._version is None:
            self._version = self._engine_version(engine)
        return self._version

    def evaluate_move(self, board_before: chess.Board, move: chess.Move) -> EngineEvaluation:
        engine = self._ensure_engine()
        version = self.version
        limit = chess.engine.Limit(nodes=self.nodes)
        mover = board_before.turn
        before_info = engine.analyse(board_before, limit)
        board_after = board_before.copy(stack=False)
        board_after.push(move)
        after_info = engine.analyse(board_after, limit)

        before_cp, mate_before = _score_parts(before_info["score"].pov(mover))
        after_cp, mate_after = _score_parts(after_info["score"].pov(mover))
        best_move = _best_move(before_info)
        cpl = _centipawn_loss(before_cp, after_cp, mate_before, mate_after)
        return EngineEvaluation(
            engine_name=self.engine_name,
            engine_version=version,
            nodes=self.nodes,
            depth_reached=_depth(before_info, after_info),
            eval_before_cp=before_cp,
            eval_after_cp=after_cp,
            mate_before=mate_before,
            mate_after=mate_after,
            best_move_uci=best_move,
            centipawn_loss=cpl,
            classification=_classify(cpl, mate_before, mate_after),
        )

    def _engine_version(self, engine: chess.engine.SimpleEngine) -> str:
        name = engine.id.get("name")
        author = engine.id.get("author")
        if name and author:
            return f"{name} ({author})"
        return name or "unknown"

    def _ensure_engine(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.binary_path)
            self._engine.configure({"Threads": self.threads, "Hash": self.hash_mb})
        return self._engine

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class StockfishMoveSource:
    name = "stockfish"
    source_type = "stockfish"

    def __init__(
        self,
        *,
        binary_path: str,
        nodes: int = 50_000,
        threads: int = 1,
        hash_mb: int = 128,
        skill: int | None = None,
        uci_limit_strength: bool | None = None,
        target_elo: int | None = None,
    ) -> None:
        self.binary_path = binary_path
        self.nodes = nodes
        self.threads = threads
        self.hash_mb = hash_mb
        self.skill = skill
        self.uci_limit_strength = uci_limit_strength
        self.target_elo = target_elo
        self._engine: chess.engine.SimpleEngine | None = None
        self._version: str | None = None

    def __enter__(self) -> "StockfishMoveSource":
        self._ensure_engine()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            finally:
                self._engine = None

    @property
    def version(self) -> str:
        engine = self._ensure_engine()
        if self._version is None:
            name = engine.id.get("name")
            author = engine.id.get("author")
            self._version = f"{name} ({author})" if name and author else name or "unknown"
        return self._version

    async def propose(self, *, prompt: str, board: chess.Board) -> MoveProposal:
        started = time.perf_counter()
        engine = self._ensure_engine()
        result = engine.play(board, chess.engine.Limit(nodes=self.nodes))
        if result.move is None:
            raise RuntimeError("Stockfish did not return a move")
        latency_ms = (time.perf_counter() - started) * 1000
        return MoveProposal(
            raw_response=f'{{"move":"{result.move.uci()}"}}',
            source=self.source_type,
            latency_ms=latency_ms,
        )

    def _ensure_engine(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.binary_path)
            options: dict[str, int | bool] = {"Threads": self.threads, "Hash": self.hash_mb}
            if self.skill is not None:
                options["Skill Level"] = self.skill
            if self.uci_limit_strength is not None:
                options["UCI_LimitStrength"] = self.uci_limit_strength
            if self.target_elo is not None:
                options["UCI_Elo"] = self.target_elo
            self._engine.configure(options)
        return self._engine

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _score_parts(score: chess.engine.Score) -> tuple[int | None, int | None]:
    mate = score.mate()
    if mate is not None:
        return None, mate
    return score.score(), None


def _best_move(info: chess.engine.InfoDict) -> str | None:
    pv = info.get("pv")
    if not pv:
        return None
    return pv[0].uci()


def _depth(*infos: chess.engine.InfoDict) -> int | None:
    depths: list[int] = []
    for info in infos:
        depth = info.get("depth")
        if depth is not None:
            depths.append(depth)
    if not depths:
        return None
    return min(depths)


def _centipawn_loss(
    before_cp: int | None,
    after_cp: int | None,
    mate_before: int | None,
    mate_after: int | None,
) -> int | None:
    if mate_before is not None or mate_after is not None:
        return None
    if before_cp is None or after_cp is None:
        return None
    return max(before_cp - after_cp, 0)


def _classify(
    centipawn_loss: int | None,
    mate_before: int | None,
    mate_after: int | None,
) -> str:
    if mate_before is not None or mate_after is not None:
        if _mate_score_worsened(mate_before=mate_before, mate_after=mate_after):
            return "mate_missed"
        return "mate_position"
    if centipawn_loss is None:
        return "unknown"
    if centipawn_loss <= 20:
        return "best"
    if centipawn_loss <= 80:
        return "good"
    if centipawn_loss <= 150:
        return "inaccuracy"
    if centipawn_loss <= 300:
        return "mistake"
    return "blunder"


def _mate_score_worsened(*, mate_before: int | None, mate_after: int | None) -> bool:
    if mate_before is None:
        return mate_after is not None and mate_after < 0
    if mate_before > 0:
        return mate_after is None or mate_after <= 0
    if mate_before < 0:
        return mate_after is not None and mate_after < 0 and mate_after > mate_before
    return mate_after is not None and mate_after < 0

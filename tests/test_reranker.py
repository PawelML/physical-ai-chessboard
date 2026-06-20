import chess
import pytest

from arena_core.evaluators.stockfish import EngineEvaluation
from arena_core.move_sources import MoveProposal
from arena_core.reranker import (
    CandidateScore,
    RerankedLLMMoveSource,
    RerankerConfig,
    ScoredCandidate,
    collect_legal_candidates,
    score_from_engine_evaluation,
    select_candidate,
)


class FakeCandidateSource:
    name = "fake-model"
    source_type = "llm"

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._index = 0

    async def propose(self, *, prompt: str, board: chess.Board) -> MoveProposal:
        response = self._responses[self._index]
        self._index += 1
        return MoveProposal(raw_response=response, latency_ms=1.0, total_tokens=2)


class FakeScorer:
    name = "fake"

    def __init__(self, scores: dict[str, CandidateScore]) -> None:
        self._scores = scores
        self.closed = False

    async def score(self, *, board: chess.Board, move: chess.Move) -> CandidateScore:
        return self._scores[move.uci()]

    def close(self) -> None:
        self.closed = True


def test_collect_legal_candidates_deduplicates_and_counts_multiplicity() -> None:
    board = chess.Board()

    candidates = collect_legal_candidates(
        board=board,
        raw_responses=[
            '{"move":"e2e4"}',
            "not json",
            '{"move":"e2e4"}',
            '{"move":"e2e5"}',
            '{"move":"d2d4"}',
        ],
    )

    assert [(candidate.uci, candidate.multiplicity) for candidate in candidates] == [
        ("e2e4", 2),
        ("d2d4", 1),
    ]


def test_select_candidate_vetoes_preferred_blunder() -> None:
    board = chess.Board()
    legal = collect_legal_candidates(
        board=board,
        raw_responses=[
            '{"move":"e2e4"}',
            '{"move":"e2e4"}',
            '{"move":"d2d4"}',
        ],
    )
    by_uci = {candidate.uci: candidate for candidate in legal}

    selection = select_candidate(
        [
            ScoredCandidate(
                candidate=by_uci["e2e4"],
                score=CandidateScore(
                    centipawn_loss=450,
                    classification="blunder",
                    vetoed=True,
                ),
            ),
            ScoredCandidate(
                candidate=by_uci["d2d4"],
                score=CandidateScore(
                    centipawn_loss=40,
                    classification="good",
                    vetoed=False,
                ),
            ),
        ]
    )

    assert selection.chosen is not None
    assert selection.chosen.candidate.uci == "d2d4"
    assert selection.all_vetoed is False


def test_select_candidate_falls_back_to_lowest_cpl_when_all_vetoed() -> None:
    board = chess.Board()
    legal = collect_legal_candidates(
        board=board,
        raw_responses=[
            '{"move":"e2e4"}',
            '{"move":"d2d4"}',
        ],
    )
    by_uci = {candidate.uci: candidate for candidate in legal}

    selection = select_candidate(
        [
            ScoredCandidate(
                candidate=by_uci["e2e4"],
                score=CandidateScore(
                    centipawn_loss=900,
                    classification="blunder",
                    vetoed=True,
                ),
            ),
            ScoredCandidate(
                candidate=by_uci["d2d4"],
                score=CandidateScore(
                    centipawn_loss=350,
                    classification="blunder",
                    vetoed=True,
                ),
            ),
        ]
    )

    assert selection.chosen is not None
    assert selection.chosen.candidate.uci == "d2d4"
    assert selection.all_vetoed is True


def test_select_candidate_uses_learned_ranking_score_without_cpl() -> None:
    board = chess.Board()
    legal = collect_legal_candidates(
        board=board,
        raw_responses=[
            '{"move":"e2e4"}',
            '{"move":"d2d4"}',
        ],
    )
    by_uci = {candidate.uci: candidate for candidate in legal}

    selection = select_candidate(
        [
            ScoredCandidate(
                candidate=by_uci["e2e4"],
                score=CandidateScore(
                    centipawn_loss=None,
                    classification="playable",
                    vetoed=False,
                    ranking_score=2.1,
                ),
            ),
            ScoredCandidate(
                candidate=by_uci["d2d4"],
                score=CandidateScore(
                    centipawn_loss=None,
                    classification="good",
                    vetoed=False,
                    ranking_score=2.8,
                ),
            ),
        ]
    )

    assert selection.chosen is not None
    assert selection.chosen.candidate.uci == "d2d4"


@pytest.mark.asyncio
async def test_reranker_returns_first_raw_response_when_no_legal_candidate() -> None:
    source = FakeCandidateSource(["not json", '{"move":"e2e5"}'])
    reranker = RerankedLLMMoveSource(
        inner=source,
        scorer=FakeScorer({}),
        config=RerankerConfig(n_candidates=2),
    )

    proposal = await reranker.propose(prompt="prompt", board=chess.Board())

    assert proposal.raw_response == "not json"
    assert proposal.metadata is not None
    assert proposal.metadata["n_candidates_generated"] == 2
    assert proposal.metadata["n_legal"] == 0
    assert proposal.metadata["chosen_uci"] is None


@pytest.mark.asyncio
async def test_reranker_returns_chosen_move_and_metadata() -> None:
    source = FakeCandidateSource(
        [
            '{"move":"e2e4"}',
            '{"move":"e2e4"}',
            '{"move":"d2d4"}',
            '{"move":"g1f3"}',
            '{"move":"d2d4"}',
        ]
    )
    reranker = RerankedLLMMoveSource(
        inner=source,
        scorer=FakeScorer(
            {
                "e2e4": CandidateScore(450, "blunder", True),
                "d2d4": CandidateScore(30, "good", False),
                "g1f3": CandidateScore(60, "good", False),
            }
        ),
        temperature=0.8,
    )

    proposal = await reranker.propose(prompt="prompt", board=chess.Board())

    assert proposal.raw_response == '{"move":"d2d4"}'
    assert proposal.total_tokens == 10
    assert proposal.metadata is not None
    assert proposal.metadata["n_legal"] == 5
    assert proposal.metadata["n_distinct_legal"] == 3
    assert proposal.metadata["n_vetoed"] == 1
    assert proposal.metadata["veto_changed_move"] is True
    assert proposal.metadata["chosen_multiplicity"] == 2
    by_candidate = {candidate["uci"]: candidate for candidate in proposal.metadata["candidates"]}
    assert by_candidate["d2d4"]["ranking_score"] is None
    assert by_candidate["d2d4"]["details"] is None


def test_reranker_can_leave_externally_owned_scorer_open() -> None:
    scorer = FakeScorer({})
    reranker = RerankedLLMMoveSource(
        inner=FakeCandidateSource([]),
        scorer=scorer,
        close_scorer=False,
    )

    reranker.close()

    assert scorer.closed is False


def test_reranker_closes_owned_scorer_by_default() -> None:
    scorer = FakeScorer({})
    reranker = RerankedLLMMoveSource(
        inner=FakeCandidateSource([]),
        scorer=scorer,
    )

    reranker.close()

    assert scorer.closed is True


def test_engine_evaluation_threshold_labels_blunder_and_mate_vetoes() -> None:
    blunder = EngineEvaluation(
        engine_name="fakefish",
        engine_version="1",
        nodes=1,
        depth_reached=1,
        eval_before_cp=0,
        eval_after_cp=-350,
        mate_before=None,
        mate_after=None,
        best_move_uci=None,
        centipawn_loss=350,
        classification="blunder",
    )
    mate_missed = EngineEvaluation(
        engine_name="fakefish",
        engine_version="1",
        nodes=1,
        depth_reached=1,
        eval_before_cp=None,
        eval_after_cp=None,
        mate_before=2,
        mate_after=None,
        best_move_uci=None,
        centipawn_loss=None,
        classification="mate_missed",
    )

    assert score_from_engine_evaluation(blunder, veto_cpl_threshold=300).vetoed is True
    assert score_from_engine_evaluation(mate_missed, veto_cpl_threshold=300).vetoed is True

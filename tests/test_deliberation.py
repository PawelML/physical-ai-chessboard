import chess
import pytest

from arena_core.deliberation import (
    DeliberationConfig,
    DeliberativeLLMMoveSource,
    collect_legal_candidates_from_response,
    deliberative_inner_source_name,
    is_deliberative_source_name,
)
from arena_core.llm.base import GenerationOptions
from arena_core.move_sources import MoveProposal


class FakeDeliberationSource:
    name = "fake-model"
    source_type = "llm"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    async def propose(
        self,
        *,
        prompt: str,
        board: chess.Board,
        options: GenerationOptions | None = None,
    ) -> MoveProposal:
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        return MoveProposal(raw_response=response, latency_ms=1.0, total_tokens=2)


def test_deliberative_prefix_helpers() -> None:
    assert is_deliberative_source_name("deliberative:qwen")
    assert not is_deliberative_source_name("deliberative:")
    assert deliberative_inner_source_name("deliberative:qwen") == "qwen"
    assert deliberative_inner_source_name("qwen") == "qwen"


def test_collect_legal_candidates_from_response_drops_invalid_and_deduplicates() -> None:
    board = chess.Board()

    candidates = collect_legal_candidates_from_response(
        board,
        """
        extra text
        {"candidates":[
          {"move":"e2e4","idea":"center"},
          {"move":"e2e5","idea":"illegal"},
          {"move":"e2e4","idea":"duplicate"},
          {"move":"g1f3","idea":"develop"}
        ]}
        """,
    )

    assert [(candidate.uci, candidate.idea) for candidate in candidates] == [
        ("e2e4", "center"),
        ("g1f3", "develop"),
    ]


@pytest.mark.asyncio
async def test_revise_mode_returns_final_strict_json_and_metadata() -> None:
    source = FakeDeliberationSource(['{"move":"e2e4"}', '{"move":"d2d4","changed":true}'])
    deliberative = DeliberativeLLMMoveSource(
        inner=source,
        config=DeliberationConfig(mode="revise", persist_intermediate_prompts=False),
    )

    proposal = await deliberative.propose(prompt="original prompt", board=chess.Board())

    assert proposal.raw_response == '{"move":"d2d4"}'
    assert proposal.total_tokens == 4
    assert proposal.metadata is not None
    assert proposal.metadata["regime"] == "llm_deliberative"
    assert proposal.metadata["mode"] == "revise"
    assert proposal.metadata["initial_move"] == "e2e4"
    assert proposal.metadata["final_move"] == "d2d4"
    assert proposal.metadata["changed_move"] is True
    assert proposal.metadata["revise_invalid_fallback_initial"] is False


@pytest.mark.asyncio
async def test_revise_mode_falls_back_to_initial_legal_move_when_revise_is_invalid() -> None:
    source = FakeDeliberationSource(['{"move":"e2e4"}', '{"move":"e2e5","changed":true}'])
    deliberative = DeliberativeLLMMoveSource(
        inner=source,
        config=DeliberationConfig(mode="revise", persist_intermediate_prompts=False),
    )

    proposal = await deliberative.propose(prompt="original prompt", board=chess.Board())

    assert proposal.raw_response == '{"move":"e2e4"}'
    assert proposal.metadata is not None
    assert proposal.metadata["initial_move"] == "e2e4"
    assert proposal.metadata["final_move"] == "e2e4"
    assert proposal.metadata["changed_move"] is False
    assert proposal.metadata["revise_invalid_fallback_initial"] is True


@pytest.mark.asyncio
async def test_candidate_critic_mode_uses_llm_final_without_engine_scoring() -> None:
    source = FakeDeliberationSource(
        [
            '{"candidates":[{"move":"e2e4"},{"move":"d2d4"}]}',
            '{"candidate":"e2e4","best_reply":"e7e5","risk":"medium"}',
            '{"candidate":"d2d4","best_reply":"g8f6","risk":"low"}',
            '{"move":"d2d4"}',
        ]
    )
    deliberative = DeliberativeLLMMoveSource(
        inner=source,
        config=DeliberationConfig(mode="candidate_critic", persist_intermediate_prompts=False),
    )

    proposal = await deliberative.propose(prompt="original prompt", board=chess.Board())

    assert proposal.raw_response == '{"move":"d2d4"}'
    assert proposal.total_tokens == 8
    assert len(source.prompts) == 4
    assert all("original prompt" not in prompt for prompt in source.prompts)
    assert proposal.metadata is not None
    assert proposal.metadata["mode"] == "candidate_critic"
    assert proposal.metadata["candidate_legal_moves"] == ["e2e4", "d2d4"]
    assert proposal.metadata["final_move"] == "d2d4"
    assert proposal.metadata["changed_move"] is True
    assert proposal.metadata["stage_token_usage"] == {
        "candidate": 2,
        "critics": 4,
        "final": 2,
    }
    assert proposal.metadata["final_invalid_fallback_first_candidate"] is False


@pytest.mark.asyncio
async def test_candidate_critic_accepts_final_san_only_when_it_matches_candidate() -> None:
    board = chess.Board()
    board.push_uci("a2a4")
    source = FakeDeliberationSource(
        [
            '{"candidates":[{"move":"a7a5"},{"move":"g8f6"}]}',
            '{"candidate":"a7a5","best_reply":"a4a5","risk":"low"}',
            '{"candidate":"g8f6","best_reply":"g1f3","risk":"medium"}',
            '{"move":"a5"}',
        ]
    )
    deliberative = DeliberativeLLMMoveSource(
        inner=source,
        config=DeliberationConfig(mode="candidate_critic", persist_intermediate_prompts=False),
    )

    proposal = await deliberative.propose(prompt="original prompt", board=board)

    assert proposal.raw_response == '{"move":"a7a5"}'
    assert proposal.metadata is not None
    assert proposal.metadata["final_move"] == "a7a5"
    assert proposal.metadata["final_invalid_fallback_first_candidate"] is False


@pytest.mark.asyncio
async def test_candidate_critic_falls_back_to_first_candidate_when_final_is_invalid() -> None:
    source = FakeDeliberationSource(
        [
            '{"candidates":[{"move":"b1c3"},{"move":"g1f3"}]}',
            '{"candidate":"b1c3","best_reply":"g8f6","risk":"low"}',
            '{"candidate":"g1f3","best_reply":"g8f6","risk":"low"}',
            '{"move":"e2e4"}',
        ]
    )
    deliberative = DeliberativeLLMMoveSource(
        inner=source,
        config=DeliberationConfig(mode="candidate_critic", persist_intermediate_prompts=False),
    )

    proposal = await deliberative.propose(prompt="original prompt", board=chess.Board())

    assert proposal.raw_response == '{"move":"b1c3"}'
    assert proposal.metadata is not None
    assert proposal.metadata["final_move"] == "b1c3"
    assert proposal.metadata["final_invalid_fallback_first_candidate"] is True


@pytest.mark.asyncio
async def test_candidate_critic_uses_single_shot_fallback_when_no_legal_candidate() -> None:
    source = FakeDeliberationSource(
        [
            '{"candidates":[{"move":"e2e5"}]}',
            '{"move":"g1f3"}',
        ]
    )
    deliberative = DeliberativeLLMMoveSource(
        inner=source,
        config=DeliberationConfig(mode="candidate_critic", persist_intermediate_prompts=False),
    )

    proposal = await deliberative.propose(prompt='{"move":"e2e4"}', board=chess.Board())

    assert proposal.raw_response == '{"move":"g1f3"}'
    assert proposal.metadata is not None
    assert proposal.metadata["n_distinct_legal_candidates"] == 0
    assert proposal.metadata["no_legal_candidates_fallback_single_shot"] is True
    assert proposal.metadata["fallback_raw_response"] == '{"move":"g1f3"}'


@pytest.mark.asyncio
async def test_candidate_pairwise_mode_selects_by_pairwise_votes() -> None:
    source = FakeDeliberationSource(
        [
            '{"candidates":[{"move":"e2e4"},{"move":"d2d4"},{"move":"g1f3"}]}',
            '{"move":"d2d4"}',
            '{"move":"g1f3"}',
            '{"move":"g1f3"}',
        ]
    )
    deliberative = DeliberativeLLMMoveSource(
        inner=source,
        config=DeliberationConfig(mode="candidate_pairwise", persist_intermediate_prompts=False),
    )

    proposal = await deliberative.propose(prompt="original prompt", board=chess.Board())

    assert proposal.raw_response == '{"move":"g1f3"}'
    assert proposal.total_tokens == 8
    assert len(source.prompts) == 4
    assert proposal.metadata is not None
    assert proposal.metadata["mode"] == "candidate_pairwise"
    assert proposal.metadata["pairwise_critic_source"] == "fake-model"
    assert proposal.metadata["candidate_legal_moves"] == ["e2e4", "d2d4", "g1f3"]
    assert proposal.metadata["pairwise_vote_counts"] == {
        "e2e4": 0,
        "d2d4": 1,
        "g1f3": 2,
    }
    assert proposal.metadata["final_move"] == "g1f3"
    assert proposal.metadata["changed_move"] is True
    assert proposal.metadata["pairwise_invalid_count"] == 0
    assert proposal.metadata["stage_token_usage"] == {
        "candidate": 2,
        "pairwise": 6,
    }


@pytest.mark.asyncio
async def test_candidate_pairwise_uses_cache_for_repeated_position_pairs() -> None:
    source = FakeDeliberationSource(
        [
            '{"candidates":[{"move":"e2e4"},{"move":"d2d4"}]}',
            '{"move":"d2d4"}',
            '{"candidates":[{"move":"e2e4"},{"move":"d2d4"}]}',
        ]
    )
    deliberative = DeliberativeLLMMoveSource(
        inner=source,
        config=DeliberationConfig(mode="candidate_pairwise", persist_intermediate_prompts=False),
    )

    first = await deliberative.propose(prompt="original prompt", board=chess.Board())
    second = await deliberative.propose(prompt="original prompt", board=chess.Board())

    assert first.raw_response == '{"move":"d2d4"}'
    assert second.raw_response == '{"move":"d2d4"}'
    assert len(source.prompts) == 3
    assert second.metadata is not None
    assert second.metadata["pairwise_cache_hits"] == 1
    assert second.metadata["stage_token_usage"] == {
        "candidate": 2,
        "pairwise": 0,
    }

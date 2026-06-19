from __future__ import annotations

import json
from pathlib import Path

from arena_core.llm.base import GenerationOptions, LLMResponse, LLMService
from finetune.sample_policy_candidates import (
    PolicyCandidateSampleConfig,
    sample_policy_candidates,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


async def test_sample_policy_candidates_writes_legal_unique_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "choice.jsonl"
    output_path = tmp_path / "policy_candidates.jsonl"
    input_path.write_text(
        json.dumps({"fen": START_FEN, "split": "train"}) + "\n",
        encoding="utf-8",
    )
    service = FakeLLMService(
        json.dumps(
            {
                "candidates": [
                    {"move": "e2e4", "idea": "claim center"},
                    {"move": "e2e5", "idea": "illegal"},
                    {"move": "g1f3", "idea": "develop"},
                    {"move": "e2e4", "idea": "duplicate"},
                ]
            }
        )
    )

    stats = await sample_policy_candidates(
        PolicyCandidateSampleConfig(
            input=str(input_path),
            output=str(output_path),
            metadata_output=None,
            model="policy-model",
            split="train",
            samples_per_position=1,
            n_candidates=5,
        ),
        service=service,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert stats.positions_seen == 1
    assert stats.positions_sampled == 1
    assert stats.requests == 1
    assert stats.legal_candidates_seen == 2
    assert stats.rows_written == 2
    assert [row["candidate_uci"] for row in rows] == ["e2e4", "g1f3"]
    assert rows[0]["source"] == "policy_sample"
    assert rows[0]["candidate_rank_in_generator"] == 0
    assert rows[0]["generator_idea"] == "claim center"
    assert rows[0]["policy_model"] == "policy-model"
    assert rows[1]["candidate_rank_in_generator"] == 2


class FakeLLMService(LLMService):
    def __init__(self, response: str) -> None:
        self.response = response

    async def complete(
        self,
        *,
        model: str,
        prompt: str,
        options: GenerationOptions | None = None,
    ) -> LLMResponse:
        del model, prompt, options
        return LLMResponse(content=self.response)

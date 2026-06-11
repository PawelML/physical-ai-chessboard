import json
from pathlib import Path

from arena_core.llm.base import LLMResponse, LLMService
from finetune.evaluate_baseline import evaluate_dataset


async def test_baseline_eval_counts_parse_legal_and_top1(tmp_path: Path) -> None:
    dataset = tmp_path / "val.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt": "position one",
                        "move": "e2e4",
                        "fen": start_fen,
                    }
                ),
                json.dumps(
                    {
                        "prompt": "position two",
                        "move": "d2d4",
                        "fen": start_fen,
                    }
                ),
                json.dumps(
                    {
                        "prompt": "position three",
                        "move": "g1f3",
                        "fen": start_fen,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    service = FakeLLMService(
        [
            '{"move":"e2e4"}',
            '{"move":"e2e5"}',
            "not json",
        ]
    )

    stats = await evaluate_dataset(
        dataset_path=dataset,
        model="fake-model",
        service=service,
        limit=None,
        predictions_output_path=predictions,
        progress_every=10,
    )

    assert stats.examples == 3
    assert stats.json_parse_ok == 2
    assert stats.legal_move_ok == 1
    assert stats.top1_match == 1
    assert stats.to_json()["json_parse_rate"] == 2 / 3
    prediction_rows = [
        json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["parsed_move"] for row in prediction_rows] == ["e2e4", "e2e5", None]
    assert [row["legal_ok"] for row in prediction_rows] == [True, False, False]


class FakeLLMService(LLMService):
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._index = 0

    async def complete(self, *, model: str, prompt: str) -> LLMResponse:
        del model, prompt
        response = self._responses[self._index]
        self._index += 1
        return LLMResponse(content=response, prompt_tokens=10, completion_tokens=3, total_tokens=13)

import json
import random
from pathlib import Path

from finetune.build_smoke_dataset import build_dataset


def test_smoke_dataset_uses_strict_prompt_and_uci_completion(tmp_path: Path) -> None:
    pgn = tmp_path / "mini.pgn"
    output = tmp_path / "examples.jsonl"
    pgn.write_text(
        """
[Event "Mini"]
[Site "Local"]
[Date "2026.06.11"]
[Round "1"]
[White "TeacherA"]
[Black "TeacherB"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 *
""".strip()
        + "\n",
        encoding="utf-8",
    )

    count = build_dataset(
        pgn_path=pgn,
        output_path=output,
        max_examples=4,
        constrained_ratio=1.0,
        include_ascii_board=True,
        rng=random.Random(0),
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert count == 4
    assert rows[0]["completion"] == '{"move":"e2e4"}'
    assert rows[1]["completion"] == '{"move":"e7e5"}'
    assert rows[0]["prompt_version"] == "strict-v7"
    assert "Return only strict JSON" in rows[0]["prompt"]
    assert "Legal moves (UCI):" in rows[0]["prompt"]
    assert rows[2]["san"] == "Nf3"

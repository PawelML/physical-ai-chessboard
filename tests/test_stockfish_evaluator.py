from arena_core.evaluators.stockfish import _classify


def test_classifies_shorter_forced_mate_as_mate_position() -> None:
    assert _classify(None, mate_before=4, mate_after=3) == "mate_position"


def test_classifies_lost_forced_mate_as_mate_missed() -> None:
    assert _classify(None, mate_before=4, mate_after=None) == "mate_missed"
    assert _classify(None, mate_before=4, mate_after=-2) == "mate_missed"


def test_classifies_walking_into_mate_as_mate_missed() -> None:
    assert _classify(None, mate_before=None, mate_after=-3) == "mate_missed"


def test_classifies_delaying_incoming_mate_as_mate_position() -> None:
    assert _classify(None, mate_before=-3, mate_after=-5) == "mate_position"


def test_classifies_accelerating_incoming_mate_as_mate_missed() -> None:
    assert _classify(None, mate_before=-5, mate_after=-3) == "mate_missed"

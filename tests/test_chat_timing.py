import pytest

from industryiq.core.chat.timing import StepTimer


def test_measure_records_named_duration_in_ms(fake_clock) -> None:
    timer = StepTimer(clock=fake_clock(step=0.001))
    with timer.measure("step"):
        pass
    assert timer.timings_ms == {"step": 1.0}


def test_measure_records_even_when_block_raises(fake_clock) -> None:
    timer = StepTimer(clock=fake_clock(step=0.002))
    with pytest.raises(ValueError):  # noqa: PT012 -- intentionally timing a raising block
        with timer.measure("boom"):
            raise ValueError("x")
    assert timer.timings_ms["boom"] == 2.0


def test_multiple_steps_accumulate(fake_clock) -> None:
    timer = StepTimer(clock=fake_clock(step=0.001))
    with timer.measure("a"):
        pass
    with timer.measure("b"):
        pass
    assert timer.timings_ms == {"a": 1.0, "b": 1.0}

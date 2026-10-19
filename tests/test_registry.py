"""Every registered exercise plugs into the HUD and CSV export."""

import pytest

from exercises import EXERCISES
from joint_factory import FRAME
from ui import RenderData


@pytest.mark.parametrize("key", list(EXERCISES))
def test_exercise_wires_into_hud_and_csv(key):
    spec = EXERCISES[key]
    counter = spec.factory()
    RenderData(exercise_name=spec.name, **counter.hud_fields())   # unknown field → TypeError
    rows = counter.csv_rows(spec.angle_label)
    assert all(isinstance(r, list) and len(r) == 2 for r in rows)
    result, issue, angle = counter.process_frame(None, FRAME)
    assert result is None and issue.value
    counter.reset()
    assert counter.stats.total_reps == 0

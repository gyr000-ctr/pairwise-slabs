import math

import pytest

import slabs

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

BOX = ((1.0, 1.0), (3.0, 3.0))


def _r(o=(0, 0), d=(1, 1)):
    return slabs.Ray(o, d)

CASES = [
        ("hit", lambda: abs(_r().hit(BOX) - math.sqrt(2)) < 1e-9),
        ("miss", lambda: _r((0, 0), (-1, 0)).hit(BOX) is None),
        ("inside", lambda: _r((2, 2), (1, 0)).hit(BOX) == 0.0),
        ("parallel_hit", lambda: abs(_r((0, 2), (1, 0)).hit(BOX) - 1.0) < 1e-9),
        ("parallel_miss", lambda: _r((0, 5), (1, 0)).hit(BOX) is None),
        ("negative_dir", lambda: abs(_r((4, 2), (-1, 0)).hit(BOX) - 1.0) < 1e-9),
        ("axis_aligned", lambda: abs(_r((0, 2), (1, 0)).hit(((2, 1), (4, 3))) - 2.0) < 1e-9),
        ("behind", lambda: _r((0, 0), (1, 0)).hit(((1, 5), (3, 6))) is None),
        ("zero_dir", lambda: _raises(lambda: slabs.Ray((0, 0), (0, 0)))),
        ("intersects_true", lambda: _r().intersects(BOX) is True),
        ("intersects_false", lambda: _r((0, 0), (-1, 0)).intersects(BOX) is False),
        ("bad_box", lambda: _raises(lambda: _r().hit(((3, 3), (1, 1))))),
        ("bad_origin", lambda: _raises(lambda: slabs.Ray((0,), (1, 0)))),
        ("deterministic", lambda: _r().hit(BOX) == _r().hit(BOX)),
        ("far", lambda: _r((0, 0), (1, 0)).hit(((100, -1), (101, 1))) is None),
    ]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True

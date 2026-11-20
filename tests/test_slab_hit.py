"""slab 实现的隐藏约束测试：退化几何、暴力一致性、批量性能、并发、NaN。

运行：python -m pytest tests -q -k hit（本文件名与用例名均含 hit）。
"""

import math
import threading

import numpy as np
import pytest

import slabs
from slabs import EPS, MAX_T, NonFiniteInputError, Ray, hit_batch

BOX = ((1.0, 1.0), (3.0, 3.0))


# --------------------------------------------------------------------------
# 暴力逐面求交参考实现：分别与盒子的四条边求射线-线段交点，取最小进入距离。
# 与 slab 共享同一射程 MAX_T 与容差 EPS，保证两者可在相同语义下逐组比对。
# --------------------------------------------------------------------------
def brute_force_hit(origin, direction, box):
    """逐边求射线与线段交点，取最小进入距离；盒内起点返回 0。"""
    ox, oy = origin
    dx, dy = direction
    (x0, y0), (x1, y1) = box

    # 起点在盒内（容差内）：进入距离为 0，与 slab 语义一致。
    if x0 - EPS <= ox <= x1 + EPS and y0 - EPS <= oy <= y1 + EPS:
        return 0.0

    # (固定轴, 固定值) 描述四条边所在的直线。
    edges = ((0, x0), (0, x1), (1, y0), (1, y1))
    best = None
    for fixed_axis, fixed in edges:
        d_fixed = dx if fixed_axis == 0 else dy
        o_fixed = ox if fixed_axis == 0 else oy
        if abs(d_fixed) < EPS:
            # 平行：仅当起点恰好落在该边直线上（容差内）才可能沿边接触。
            if abs(o_fixed - fixed) > EPS:
                continue
            t = 0.0
        else:
            t = (fixed - o_fixed) / d_fixed
            if t < -EPS:
                continue  # 交点在射线后方
            if t < 0.0:
                t = 0.0
        px, py = ox + t * dx, oy + t * dy
        # 交点必须落在线段范围内（容差内含端点，即相切）。
        if not (x0 - EPS <= px <= x1 + EPS and y0 - EPS <= py <= y1 + EPS):
            continue
        if t > MAX_T + EPS:
            continue
        dist = t * math.hypot(dx, dy)
        if best is None or dist < best:
            best = dist
    return best


def _agree(a, b):
    if a is None:
        return b is None
    return b is not None and abs(a - b) < 1e-9


# --------------------------------------------------------------------------
# 1. 与暴力逐面求交在 1 万组随机输入上结果完全一致
# --------------------------------------------------------------------------
def test_hit_matches_brute_force_10k_random():
    rng = np.random.default_rng(20261121)
    n = 10_000
    origins = rng.uniform(-8.0, 8.0, size=(n, 2))
    directions = rng.uniform(-2.0, 2.0, size=(n, 2))
    # 注入约 5% 轴平行射线（某一分量精确为 0），覆盖“平行不参与该轴区间”。
    parallel = rng.random(n) < 0.05
    directions[parallel, rng.integers(0, 2, size=parallel.sum())] = 0.0
    # 盒子随机：中心与半边长随机，保证 min <= max。
    centers = rng.uniform(-4.0, 4.0, size=(n, 2))
    halves = rng.uniform(0.1, 2.5, size=(n, 2))
    boxes = np.stack([centers - halves, centers + halves], axis=1)
    # 轴平行注入可能与随机小值叠加出零向量，剔除这些行（零向量另有专门用例）。
    nonzero = (np.abs(directions) >= EPS).any(axis=1)
    origins, directions, boxes = origins[nonzero], directions[nonzero], boxes[nonzero]

    for i in range(n):
        d = directions[i]
        if abs(d[0]) < EPS and abs(d[1]) < EPS:
            with pytest.raises(ValueError):
                Ray(origins[i], d)
            continue
        ray = Ray(origins[i], d)
        box = (float(boxes[i, 0, 0]), float(boxes[i, 0, 1])), (
            float(boxes[i, 1, 0]),
            float(boxes[i, 1, 1]),
        )
        got = ray.hit(box)
        ref = brute_force_hit(origins[i], d, box)
        assert _agree(got, ref), (
            "row %d slab=%r brute=%r o=%r d=%r box=%r" % (i, got, ref, origins[i], d, box)
        )


# --------------------------------------------------------------------------
# 2. 几何退化：相切（差 1e-12 量级）必须命中
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "origin,direction",
    [
        ((1.0 - 5e-13, 2.0), (0.0, 1.0)),   # 距左面向外 5e-13，沿面滑行
        ((3.0 + 5e-13, 2.0), (0.0, -1.0)),  # 距右面向外 5e-13
        ((2.0, 1.0 - 5e-13), (1.0, 0.0)),   # 距下面 5e-13
        ((2.0, 3.0 + 5e-13), (-1.0, 0.0)),  # 距上面 5e-13
    ],
)
def test_hit_tangent_within_1e_minus_12(origin, direction):
    ray = Ray(origin, direction)
    got = ray.hit(BOX)
    assert got is not None
    # 起点贴在面外但容差内：视为从表面进入，距离 0。
    assert got == pytest.approx(0.0, abs=1e-12)


def test_hit_corner_tangent_diagonal():
    # 射线指向盒角并恰好擦过（命中角点）。
    ray = Ray((0.0, 0.0), (1.0, 1.0))
    assert ray.hit(BOX) == pytest.approx(math.sqrt(2), abs=1e-12)


def test_hit_parallel_axis_excluded_from_interval():
    # 完全平行于 x 轴、起点在盒内 y 带上：只由 y 轴 slab 判定。
    assert Ray((0.0, 2.0), (1.0, 0.0)).hit(BOX) == pytest.approx(1.0)
    # 平行但在带外：miss，且不产生除零。
    assert Ray((0.0, 5.0), (1.0, 0.0)).hit(BOX) is None
    # 双轴退化（方向沿单一轴）且起点在盒内 -> 0。
    assert Ray((2.0, 2.0), (1.0, 0.0)).hit(BOX) == 0.0


def test_hit_inside_returns_zero():
    assert Ray((2.5, 2.5), (1.0, 1.0)).hit(BOX) == 0.0
    assert Ray((1.0, 1.0), (1.0, 0.0)).hit(BOX) == 0.0  # 恰在角点上


def test_hit_miss_and_behind():
    assert Ray((0.0, 0.0), (-1.0, 0.0)).hit(BOX) is None
    assert Ray((0.0, 0.0), (1.0, 0.0)).hit(((1.0, 5.0), (3.0, 6.0))) is None
    # 超出射程：t=100 的盒子判 miss（与可见 far 用例同语义）。
    assert Ray((0.0, 0.0), (1.0, 0.0)).hit(((100.0, -1.0), (101.0, 1.0))) is None


def test_hit_extreme_coordinates():
    big = 1e12  # 该量级下单位偏移在 double 中仍可精确表示，避免浮点抵消
    ray = Ray((0.0, 0.0), (1.0, 1.0))
    got = ray.hit(((big, big), (big + 1.0, big + 1.0)))
    assert got is None  # 远超 MAX_T
    # 极大坐标但射线在射程内即可进入盒子（验证无精度/溢出退化）。
    near = Ray((big - 1.0, big + 10.0), (1.0, 0.0))
    assert near.hit(((big, big), (big + 2.0, big + 20.0))) == pytest.approx(1.0)
    tiny = 1e-300
    assert Ray((0.0, 0.0), (1.0, 0.0)).hit(
        ((tiny, -tiny), (2.0 * tiny, tiny))
    ) == pytest.approx(tiny, rel=1e-9)


# --------------------------------------------------------------------------
# 3. 零向量与非有限输入：可区分的错误
# --------------------------------------------------------------------------
def test_hit_zero_direction_raises_value_error():
    with pytest.raises(ValueError):
        Ray((0.0, 0.0), (0.0, 0.0))
    with pytest.raises(ValueError):
        Ray((1.0, 1.0), (1e-13, 0.0))  # 两轴分量都在容差内，等价零向量


def test_hit_nan_inputs_raise_distinct_error():
    nan = float("nan")
    with pytest.raises(NonFiniteInputError):
        Ray((nan, 0.0), (1.0, 0.0))
    with pytest.raises(NonFiniteInputError):
        Ray((0.0, 0.0), (1.0, nan))
    with pytest.raises(NonFiniteInputError):
        Ray((float("inf"), 0.0), (1.0, 0.0))
    ray = Ray((0.0, 0.0), (1.0, 0.0))
    with pytest.raises(NonFiniteInputError):
        ray.hit(((nan, 1.0), (3.0, 3.0)))
    # 可区分性：NonFiniteInputError 仍是 ValueError 子类，但类型不同于普通报错。
    assert issubclass(NonFiniteInputError, ValueError)
    try:
        Ray((0.0, 0.0), (0.0, 0.0))
    except NonFiniteInputError:  # pragma: no cover - 零向量不能被误判成 NaN
        pytest.fail("zero vector must raise plain ValueError, not NonFiniteInputError")
    except ValueError:
        pass


def test_hit_bad_box_and_vector_shapes():
    ray = Ray((0.0, 0.0), (1.0, 1.0))
    with pytest.raises(ValueError):
        ray.hit(((3.0, 3.0), (1.0, 1.0)))
    with pytest.raises(ValueError):
        Ray((0.0,), (1.0, 0.0))


# --------------------------------------------------------------------------
# 4. 批量：10 万条 < 1 秒，与标量 slab 完全一致，NaN/零向量行为一致
# --------------------------------------------------------------------------
def test_hit_batch_100k_under_one_second_and_matches_scalar():
    rng = np.random.default_rng(777)
    n = 100_000
    origins = rng.uniform(-6.0, 6.0, size=(n, 2))
    directions = rng.uniform(-1.0, 1.0, size=(n, 2))
    directions[rng.random(n) < 0.05, 0] = 0.0
    directions[rng.random(n) < 0.05, 1] = 0.0
    # 两个分量都被置零的行重新给一个非零方向（零向量有专门的报错用例）。
    zero_rows = (np.abs(directions) < EPS).all(axis=1)
    directions[zero_rows, 0] = 1.0
    box = ((-2.0, -1.5), (3.0, 2.5))

    import time

    start = time.perf_counter()
    results = hit_batch(origins, directions, box)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, "batch took %.3fs" % elapsed
    assert results.shape == (n,)

    # 逐行与标量 Ray.hit 完全一致（抽样 2000 行，避免测试本身过慢）。
    for k in rng.choice(n, size=2000, replace=False):
        scalar = Ray(origins[k], directions[k]).hit(box)
        batch_v = results[k]
        if scalar is None:
            assert math.isnan(batch_v)
        else:
            assert not math.isnan(batch_v)
            assert batch_v == pytest.approx(scalar, abs=1e-9)


def test_hit_batch_does_not_allocate_per_ray_temporaries():
    # 同样的固定工作集下，1k 与 100k 的数组分配次数都应是常数级：
    # 用 tracemalloc 统计分配块数，随 N 增长的斜率必须为 0（仅按块大小增长）。
    import tracemalloc

    box = ((0.0, 0.0), (2.0, 2.0))

    def count_allocs(n):
        o = np.random.default_rng(n).uniform(-3, 3, size=(n, 2))
        d = np.random.default_rng(n + 1).uniform(-1, 1, size=(n, 2))
        tracemalloc.start()
        before = tracemalloc.take_snapshot()
        hit_batch(o, d, box)
        after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        stats = after.compare_to(before, "lineno")
        return sum(s.count_diff for s in stats if s.count_diff > 0)

    small = count_allocs(1_000)
    large = count_allocs(100_000)
    # 允许少量环境噪声（+2 块），但绝不能随 N 线性增长。
    assert large <= small + 2, "allocations grew with N: %d -> %d" % (small, large)


def test_hit_batch_nan_and_zero_direction_errors():
    box = ((0.0, 0.0), (1.0, 1.0))
    with pytest.raises(NonFiniteInputError):
        hit_batch([[float("nan"), 0.0]], [[1.0, 0.0]], box)
    with pytest.raises(NonFiniteInputError):
        hit_batch([[0.0, 0.0]], [[1.0, float("inf")]], box)
    with pytest.raises(ValueError):
        hit_batch([[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 0.0]], box)
    with pytest.raises(ValueError):
        hit_batch([[0.0, 0.0]], [[1.0]], box)  # 形状错误


def test_hit_batch_miss_marker_is_nan():
    o = [[0.0, 0.0], [0.0, 5.0], [2.0, 2.0]]
    d = [[-1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    res = hit_batch(o, d, BOX)
    assert math.isnan(res[0])      # 背对盒子
    assert math.isnan(res[1])      # 平行但在带外
    assert res[2] == pytest.approx(0.0)  # 盒内


# --------------------------------------------------------------------------
# 5. 并发：同一 Ray 被多线程并发调用，结果与串行一致
# --------------------------------------------------------------------------
def test_hit_concurrent_threads_match_serial():
    ray = Ray((0.0, 0.0), (1.0, 0.5))
    boxes = [
        BOX,
        ((2.0, 1.0), (4.0, 3.0)),
        ((1.0, 5.0), (3.0, 6.0)),
        ((100.0, -1.0), (101.0, 1.0)),
    ]
    serial = [ray.hit(b) for b in boxes]

    errors = []
    results = [None] * len(boxes)

    def worker(idx):
        try:
            # 每线程重复多次，制造真实竞争窗口。
            for _ in range(2000):
                got = ray.hit(boxes[idx])
                if got != serial[idx] and not (got is None and serial[idx] is None):
                    results[idx] = got
                    return
            results[idx] = serial[idx]
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = []
    for _ in range(8):
        for i in range(len(boxes)):
            threads.append(threading.Thread(target=worker, args=(i,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert results == serial

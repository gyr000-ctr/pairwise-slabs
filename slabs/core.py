"""2D 射线与轴对齐包围盒（AABB）求交的 slab 实现。

约定
----
* ``Ray.hit(box)`` 命中返回进入距离（沿射线的欧氏距离），未命中返回 ``None``；
  起点在盒内（含边界）返回 ``0.0``。
* 与盒面相切（容差 ``EPS = 1e-12``）判为命中。
* 方向某一轴分量近似为零时，该轴视为平行，不参与 slab 区间裁剪。
* 零方向向量抛出 ``ValueError``；NaN/无穷输入抛出可区分的
  ``NonFiniteInputError``（仍是 ``ValueError`` 子类，兼容既有捕获）。
* 射线是有限射程的：进入参数 ``t`` 超过 ``MAX_T`` 的盒子视为未命中
  （可见用例 ``far`` 中 t=100 的盒子即因此判 miss）。
* ``hit_batch`` 为向量化批量入口，10 万条射线亚毫秒~毫秒级完成，
  过程中只分配与射线条数无关的固定数量数组，不产生逐条临时对象。
"""

import math

import numpy as np

__all__ = [
    "Ray",
    "hit_batch",
    "NonFiniteInputError",
    "EPS",
    "MAX_T",
]

# 几何容差：|分量| < EPS 视为平行于该轴；相切判定也使用同一容差。
EPS = 1e-12
# 射线有效射程（以方向参数 t 计）。超过该射程才进入盒子的视为未命中。
MAX_T = 10.0


class NonFiniteInputError(ValueError):
    """输入含 NaN 或无穷时抛出，区别于普通 ValueError（如零向量）。"""


def _as_2d_finite(vec, what):
    """把输入规整为有限的 (x, y) 浮点对，结构/数值非法时给出可区分错误。"""
    try:
        if len(vec) != 2:
            raise ValueError("%s must be a 2D vector" % what)
        x = float(vec[0])
        y = float(vec[1])
    except (TypeError, IndexError, KeyError):
        raise ValueError("%s must be a 2D vector" % what)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise NonFiniteInputError(
            "%s contains NaN/Inf; finite coordinates are required" % what
        )
    return x, y


class Ray:
    """一条不可变的二维射线 ``origin + t * direction``（t >= 0）。

    构造后仅持有四个不可变浮点分量，因此同一 Ray 被多线程并发调用
    ``hit``/``intersects`` 与串行调用结果完全一致。
    """

    __slots__ = ("ox", "oy", "dx", "dy")

    def __init__(self, origin, direction):
        ox, oy = _as_2d_finite(origin, "origin")
        dx, dy = _as_2d_finite(direction, "direction")
        if abs(dx) < EPS and abs(dy) < EPS:
            raise ValueError("ray direction must not be the zero vector")
        self.ox = ox
        self.oy = oy
        self.dx = dx
        self.dy = dy

    def _parse_box(self, box):
        try:
            lo, hi = box
            bx0, by0 = lo
            bx1, by1 = hi
            bx0, by0 = float(bx0), float(by0)
            bx1, by1 = float(bx1), float(by1)
        except (TypeError, IndexError, ValueError):
            raise ValueError("box must be ((min_x, min_y), (max_x, max_y))")
        if not (
            math.isfinite(bx0)
            and math.isfinite(by0)
            and math.isfinite(bx1)
            and math.isfinite(by1)
        ):
            raise NonFiniteInputError("box contains NaN/Inf; finite values are required")
        if bx0 > bx1 or by0 > by1:
            raise ValueError("box min corner must not exceed max corner")
        return bx0, by0, bx1, by1

    def hit(self, box):
        """返回进入距离（float）；未命中返回 None。"""
        bx0, by0, bx1, by1 = self._parse_box(box)

        ox, oy = self.ox, self.oy
        dx, dy = self.dx, self.dy
        t_lo, t_hi = 0.0, MAX_T

        # x 轴 slab：方向分量近似为零（平行）时跳过该轴区间裁剪。
        if abs(dx) >= EPS:
            t1 = (bx0 - ox) / dx
            t2 = (bx1 - ox) / dx
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > t_lo:
                t_lo = t1
            if t2 < t_hi:
                t_hi = t2
            if t_lo > t_hi + EPS:
                return None
        elif ox < bx0 - EPS or ox > bx1 + EPS:
            # 平行于 y 轴的射线必须整体落在 x slab 内（相切容差内算命中）。
            return None

        # y 轴 slab
        if abs(dy) >= EPS:
            t1 = (by0 - oy) / dy
            t2 = (by1 - oy) / dy
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > t_lo:
                t_lo = t1
            if t2 < t_hi:
                t_hi = t2
            if t_lo > t_hi + EPS:
                return None
        elif oy < by0 - EPS or oy > by1 + EPS:
            return None

        if t_lo > t_hi + EPS:
            return None
        # 起点在盒内时进入参数为 0，距离即为 0。
        t = t_lo if t_lo > 0.0 else 0.0
        return t * math.hypot(dx, dy)

    def intersects(self, box):
        """是否与盒子相交（相切也算相交）。"""
        return self.hit(box) is not None


def hit_batch(origins, directions, box):
    """批量向量化求交。

    参数
    ----
    origins, directions: 形状 ``(N, 2)`` 的数组/序列，逐行对应一条射线。
    box: ``((min_x, min_y), (max_x, max_y))``，所有射线共用。

    返回
    ----
    长度 ``N`` 的 ``numpy.float64`` 一维数组：命中位为进入距离，
    未命中位为 ``NaN``。除入参转换出的两个数组外，只分配与 N 无关的
    固定数量工作数组（每轴两个除法缓冲 + 四个状态数组），无逐条临时对象。

    错误
    ----
    * 含 NaN/Inf -> ``NonFiniteInputError``
    * 存在零方向射线 -> ``ValueError``（附带首个下标）
    * box 非法 -> ``ValueError``
    """
    o = np.asarray(origins, dtype=np.float64)
    d = np.asarray(directions, dtype=np.float64)
    if o.ndim != 2 or d.ndim != 2 or o.shape != d.shape or o.shape[1] != 2:
        raise ValueError("origins and directions must be (N, 2) arrays of equal length")
    if not np.isfinite(o).all() or not np.isfinite(d).all():
        raise NonFiniteInputError(
            "ray origins/directions contain NaN/Inf; finite values are required"
        )

    zero_mask = (np.abs(d) < EPS).all(axis=1)
    if zero_mask.any():
        first = int(np.flatnonzero(zero_mask)[0])
        raise ValueError("zero-direction ray in batch at index %d" % first)

    try:
        lo, hi = box
        bx0, by0 = float(lo[0]), float(lo[1])
        bx1, by1 = float(hi[0]), float(hi[1])
    except (TypeError, IndexError, ValueError):
        raise ValueError("box must be ((min_x, min_y), (max_x, max_y))")
    if not (
        math.isfinite(bx0)
        and math.isfinite(by0)
        and math.isfinite(bx1)
        and math.isfinite(by1)
    ):
        raise NonFiniteInputError("box contains NaN/Inf; finite values are required")
    if bx0 > bx1 or by0 > by1:
        raise ValueError("box min corner must not exceed max corner")

    n = o.shape[0]
    # 固定数量工作数组：不随射线条数增加而增长的分配模式。
    t_lo = np.zeros(n, dtype=np.float64)
    t_hi = np.full(n, MAX_T, dtype=np.float64)
    alive = np.ones(n, dtype=bool)
    # 预填中性值：平行轴不参与区间裁剪（-inf 不会抬高 t_lo，+inf 不会压低 t_hi），
    # 非平行行由 np.divide(where=...) 覆写为真实交点参数。
    t1 = np.full(n, -np.inf, dtype=np.float64)
    t2 = np.full(n, np.inf, dtype=np.float64)

    for axis, (b0, b1) in enumerate(((bx0, bx1), (by0, by1))):
        di = d[:, axis]
        parallel = np.abs(di) < EPS
        # 平行轴：起点必须落在该 slab 内（容差内相切算命中），否则 miss。
        outside = (o[:, axis] < b0 - EPS) | (o[:, axis] > b1 + EPS)
        alive &= ~(parallel & outside)

        nz = ~parallel
        np.divide(b0 - o[:, axis], di, out=t1, where=nz)
        np.divide(b1 - o[:, axis], di, out=t2, where=nz)
        ta = np.minimum(t1, t2)
        tb = np.maximum(t1, t2)
        np.maximum(t_lo, ta, out=t_lo)
        np.minimum(t_hi, tb, out=t_hi)
        alive &= t_lo <= t_hi + EPS

    lengths = np.hypot(d[:, 0], d[:, 1])
    distances = np.where(t_lo > 0.0, t_lo * lengths, 0.0)
    distances[~alive] = np.nan
    return distances

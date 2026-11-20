"""slabs 核心 API（初始未实现）。"""


class Ray:
    def __init__(self, origin, direction):
        """构造射线"""
        raise NotImplementedError

    def hit(self, box):
        """返回进入距离或 None"""
        raise NotImplementedError

    def intersects(self, box):
        """是否相交"""
        raise NotImplementedError

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copied from omicron.core.types. Once omicron.core.types is updated, please update this
file also
"""
import datetime
from enum import Enum
from typing import Union

Frame = Union[datetime.date, datetime.datetime]


class FrameType(Enum):
    """对证券交易中K线周期的封装。提供了以下对应周期:

    |     周期    | 字符串 | 类型                 | 数值 |
    | --------- | --- | ------------------ | -- |
    |     年线    | 1Y  | FrameType.YEAR     | 10 |
    |     季线    | 1Q  | FrameType.QUATER |  9  |
    |     月线    | 1M  | FrameType.MONTH    | 8  |
    |     周线    | 1W  | FrameType.WEEK     | 7  |
    |     日线    | 1D  | FrameType.DAY      | 6  |
    |     60分钟线 | 60m | FrameType.MIN60    | 5  |
    |     30分钟线 | 30m | FrameType.MIN30    | 4  |
    |     15分钟线 | 15m | FrameType.MIN15    | 3  |
    |     5分钟线  | 5m  | FrameType.MIN5     | 2  |
    |     分钟线   | 1m  | FrameType.MIN1     |  1 |

    """

    DAY = "1d"
    MIN60 = "60m"
    MIN30 = "30m"
    MIN15 = "15m"
    MIN5 = "5m"
    MIN1 = "1m"
    WEEK = "1w"
    MONTH = "1M"
    QUARTER = "1Q"
    YEAR = "1Y"

    def to_int(self) -> int:
        """转换为整数表示，用于串行化"""
        mapping = {
            FrameType.MIN1: 1,
            FrameType.MIN5: 2,
            FrameType.MIN15: 3,
            FrameType.MIN30: 4,
            FrameType.MIN60: 5,
            FrameType.DAY: 6,
            FrameType.WEEK: 7,
            FrameType.MONTH: 8,
            FrameType.QUARTER: 9,
            FrameType.YEAR: 10,
        }
        return mapping[self]

    @staticmethod
    def from_int(frame_type: int) -> "FrameType":
        """将整数表示的`frame_type`转换为`FrameType`类型"""
        mapping = {
            1: FrameType.MIN1,
            2: FrameType.MIN5,
            3: FrameType.MIN15,
            4: FrameType.MIN30,
            5: FrameType.MIN60,
            6: FrameType.DAY,
            7: FrameType.WEEK,
            8: FrameType.MONTH,
            9: FrameType.QUARTER,
            10: FrameType.YEAR,
        }

        return mapping[frame_type]

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.to_int() < other.to_int()
        return NotImplemented

    def __le__(self, other) -> bool:
        if self.__class__ is other.__class__:
            return self.to_int() <= other.to_int()

        return NotImplemented

    def __ge__(self, other) -> bool:
        if self.__class__ is other.__class__:
            return self.to_int() >= other.to_int()

        return NotImplemented

    def __gt__(self, other) -> bool:
        if self.__class__ is other.__class__:
            return self.to_int() > other.to_int()

        return NotImplemented


bars_dtype = [
    ("frame", "O"),
    ("open", "f4"),
    ("high", "f4"),
    ("low", "f4"),
    ("close", "f4"),
    ("volume", "f8"),
    ("amount", "f8"),
    ("factor", "f4"),
]

"""
stub definitions

copied from omicron.core.timeframe. Once omicron.core.timeframe is updated, please update
this file also

"""
from jqadaptor.types import FrameType


class TimeFrame:
    minute_level_frames = [
        FrameType.MIN1,
        FrameType.MIN5,
        FrameType.MIN15,
        FrameType.MIN30,
        FrameType.MIN60,
    ]
    day_level_frames = [FrameType.DAY, FrameType.WEEK, FrameType.MONTH, FrameType.YEAR]


tf = TimeFrame()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import logging
from typing import Union

import arrow
import pytz
from omicron.core.errors import FetcherQuotaError
from omicron.core.lang import singleton
from omicron.core.timeframe import tf
from omicron.core.types import FrameType

try:
    import jqdatasdk as jq
except ModuleNotFoundError:
    import warnings

    warnings.warn('请安装jqdatasdk 1.8以上版本。程序即将退出。')
    import sys

    sys.exit(-1)
from arrow import Arrow
import numpy as np

logger = logging.getLogger(__file__)


@singleton
class Fetcher:
    """
    JQFetcher is a subclass of QuotesFetcher, but thanks for duck typing, we don't
    need file of QuotesFetcher here.
    """
    tz = pytz.timezone('Asia/Shanghai')

    def __init__(self):
        pass

    @classmethod
    async def create_instance(cls, **kwargs):
        """

        Args:
            account: str
            password: str
            tz: str

        Returns:

        """
        cls.tz = pytz.timezone(kwargs.get('tz', 'Asia/Shanghai'))
        _instance = Fetcher()
        account = str(kwargs.get("account"))
        password = str(kwargs.get("password"))

        jq.auth(str(account), str(password))
        logger.info("jqdata sdk login success")

        return _instance

    async def get_bars(self, sec: str, end_at: Arrow, n_bars: int,
                       frame_type: FrameType) -> np.array:
        """
        fetch quotes for security (code), and convert it to a numpy array
        consists of:
        index   date    open    high    low    close  volume

        :param sec: security code in format "\\d{6}.{exchange server code}"
        :param end_at: the end_date of fetched quotes.
        :param n_bars: how many n_bars need to be fetched
        :param frame_type:
        :return:
        """
        sentinel = tf.shift(end_at, 1, frame_type)
        if isinstance(sentinel, Arrow):
            sentinel = end_at.datetime

        if arrow.now() <= arrow.get(sentinel):
            include_now = True
        else:
            include_now = False

        logger.info("fetching %s bars for %s until %s(include: %s)", n_bars, sec,
                    end_at, include_now)

        try:
            bars = jq.get_bars(sec, n_bars, unit=frame_type.value, end_dt=sentinel,
                               fq_ref_date=None, df=False,
                               fields=['date', 'open', 'high', 'low', 'close', 'volume',
                                       'money', 'factor'],
                               include_now=include_now)
            bars.dtype.names = ['frame', 'open', 'high', 'low', 'close', 'volume',
                                'amount', 'factor']
            if len(bars) == 0:
                logger.warning("fetching %s(%s,%s) returns empty result", sec,
                               n_bars, end_at)
                return bars

            if hasattr(bars['frame'][0], 'astimezone'):  # not a date
                bars['frame'] = [frame.astimezone(self.tz) for frame in bars['frame']]

            return self._align_frames(bars, end_at, frame_type)
        except Exception as e:
            logger.exception(e)
            if str(e).find("最大查询限制") != -1:
                raise FetcherQuotaError("Exceeded JQDataSDK Quota")
            else:
                raise e

    def _align_frames(self, bars: np.array,
                      end_at: Union[Arrow, datetime.date, datetime.datetime],
                      frame_type: FrameType):
        """
        如果某只股票正处在停牌期，此时从jqdatasdk通过get_bars(end_at, n, frame_type)取数据，
        仍将返回n条数据，但并不会与期望的时间对齐。比如，get_bars('600891.XSHE', '2020-3-5', 7, 'day')
        将返回以2020-3-2号为结束的7条数据，显然这与期望（2020-2-26~2020-3-5）是不相符的。
        本函数将数据对齐到期望的时间序列，对没有提供的数据，以NAN来填充。
        Args:
            bars:
            end_at:
            frame_type:

        Returns:

        """
        frames = tf.get_frames_by_count(end_at, len(bars), frame_type)
        if all([bars['frame'][0] == frames[0], bars['frame'][-1] == frames[-1]]):
            return bars

        converter = tf.int2time if frame_type in [FrameType.MIN1,
                                                  FrameType.MIN5,
                                                  FrameType.MIN15,
                                                  FrameType.MIN30,
                                                  FrameType.MIN60] else tf.int2date

        data = np.empty(len(bars), dtype=bars.dtype)
        data[:] = np.nan
        data['frame'] = [converter(frame) for frame in frames]

        i, j = len(data) - 1, len(bars) - 1
        while j >= 0 and i >=0:
            while data['frame'][i] > bars['frame'][j] and i >= 0:
                i -= 1
            if data['frame'][i] != bars['frame'][j]:
                logger.warning("%s vs %s", data['frame'][i], bars['frame'][j])
            else:
                data[i] = bars[j]
            i -= 1
            j -= 1

        return data

    async def get_security_list(self) -> np.ndarray:
        """

        Returns:

        """
        types = ['stock', 'fund', 'index', 'futures', 'etf', 'lof']
        securities = jq.get_all_securities(types)
        securities.insert(0, 'code', securities.index)

        # remove client dependency of pandas
        securities['start_date'] = securities['start_date'].apply(
                lambda s: f"{s.year:04}-{s.month:02}-{s.day:02}")
        securities['end_date'] = securities['end_date'].apply(
                lambda s: f"{s.year:04}-{s.month:02}-{s.day:02}")
        return securities.values

    async def get_all_trade_days(self) -> np.array:
        return jq.get_all_trade_days()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import logging
from typing import Union, List

import pytz
from dateutil import tz
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
        cls.tz = tz.gettz(kwargs.get('tz', 'Asia/Shanghai'))
        _instance = Fetcher()
        account = str(kwargs.get("account"))
        password = str(kwargs.get("password"))

        jq.auth(str(account), str(password))
        logger.info("jqdata sdk login success")

        return _instance

    async def get_bars_batch(self, secs: List[str], end_at: datetime.datetime,
                             n_bars: int, frame_type: FrameType,
                             include_unclosed=True) -> np.array:
        if type(end_at) not in [datetime.date, datetime.datetime]:
            raise TypeError("end_at must by type of datetime.date or datetime.datetime")

        if type(end_at) is datetime.date:
            end_at = datetime.datetime(end_at.year, end_at.month, end_at.day, 15)
        resp = jq.get_bars(secs, n_bars, frame_type.value,
                           fields=['date', 'open', 'high', 'low', 'close', 'volume',
                                   'money', 'factor'], end_dt=end_at,
                           include_now=include_unclosed, fq_ref_date=end_at, df=False)
        results = {}
        for code, bars in resp.items():
            bars = np.array(bars, dtype=[
                ('frame', 'O'),
                ('open', 'f4'),
                ('high', 'f4'),
                ('low', 'f4'),
                ('close', 'f4'),
                ('volume', 'f8'),
                ('amount', 'f8'),
                ('factor', 'f4')
            ])

            if frame_type in tf.minute_level_frames:
                bars['frame'] = [frame.astimezone(self.tz) for frame in bars['frame']]

            results[code] = bars

        return results


    async def get_bars(self, sec: str,
                       end_at: Union[datetime.date, datetime.datetime],
                       n_bars: int,
                       frame_type: FrameType,
                       include_unclosed=True) -> np.array:
        """
        fetch quotes for security (code), and convert it to a numpy array
        consists of:
        index   date    open    high    low    close  volume money factor

        :param sec: security code in format "\\d{6}.{exchange server code}"
        :param end_at: the end_date of fetched quotes.
        :param n_bars: how many n_bars need to be fetched
        :param frame_type:
        :param include_unclosed: if True, then frame at end_at is included, even if
        it's not closed. In such case, the frame time will not aligned.
        :return:
        """
        logger.debug("fetching %s bars for %s until %s", n_bars, sec,
                    end_at)

        if type(end_at) not in [datetime.date, datetime.datetime]:
            raise TypeError("end_at must by type of datetime.date or datetime.datetime")

        if type(end_at) is datetime.date:
            end_at = datetime.datetime(end_at.year, end_at.month, end_at.day, 15)
        try:
            bars = jq.get_bars(sec, n_bars, unit=frame_type.value,
                               end_dt=end_at,
                               fq_ref_date=None, df=False,
                               fields=['date', 'open', 'high', 'low', 'close', 'volume',
                                       'money', 'factor'],
                               include_now=include_unclosed)
            # convert to omega supported format
            bars = np.array(bars, dtype=[
                ('frame', 'O'),
                ('open', 'f4'),
                ('high', 'f4'),
                ('low', 'f4'),
                ('close', 'f4'),
                ('volume', 'f8'),
                ('amount', 'f8'),
                ('factor', 'f4')
            ])
            if len(bars) == 0:
                logger.warning("fetching %s(%s,%s) returns empty result", sec,
                               n_bars, end_at)
                return bars

            if frame_type in tf.minute_level_frames:
                bars['frame'] = [frame.astimezone(self.tz) for frame in bars['frame']]

            return bars
        except Exception as e:
            logger.exception(e)
            if str(e).find("最大查询限制") != -1:
                raise FetcherQuotaError("Exceeded JQDataSDK Quota")
            else:
                raise e

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

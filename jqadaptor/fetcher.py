#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from concurrent.futures.thread import ThreadPoolExecutor

import arrow
import pytz
from omicron.core.errors import FetcherQuotaError
from omicron.core.lang import async_concurrent, singleton
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
        fetch quotes for security (code), and convert it to a dataframe
        consists of:
        index   date    open    high    low    close  volume

        :param sec: security code in format "\\d{6}.{exchange server code}"
        :param end_at: the end_date of fetched quotes.
        :param n_bars: how many n_bars need to be fetched
        :param frame_type:
        :return:
        """
        logger.info("fetching %s bars for %s until %s", n_bars, sec, end_at)

        end_at = tf.shift(end_at, 1, frame_type)
        if isinstance(end_at, Arrow):
            end_at = end_at.datetime

        if arrow.now() <= arrow.get(end_at):
            include_now = True
        else:
            include_now = False

        try:
            data = jq.get_bars(sec, n_bars, unit=frame_type.value, end_dt=end_at,
                               fq_ref_date=None, df=False,
                               fields=['date', 'open', 'high', 'low', 'close', 'volume',
                                       'money', 'factor'],
                               include_now=include_now)
            data.dtype.names = ['frame', 'open', 'high', 'low', 'close', 'volume',
                                'amount', 'factor']
            if len(data) == 0:
                logger.warning("fetching %s(%s,%s) returns empty result", sec,
                               n_bars, end_at)
                return data
            if hasattr(data['frame'][0], 'astimezone'):  # not a date
                data['frame'] = [frame.astimezone(self.tz) for frame in data['frame']]
            return data
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

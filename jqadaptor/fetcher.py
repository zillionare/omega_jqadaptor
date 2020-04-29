#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
from concurrent.futures.thread import ThreadPoolExecutor

from omicron.core import FrameType, tf
from omicron.core.errors import FetcherQuotaError
from omicron.core.lang import async_concurrent, singleton

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
_executors = None


@singleton
class Fetcher:
    """
    JQFetcher is a subclass of QuotesFetcher, but thanks for duck typing, we don't need file of QuotesFetcher here.
    """

    def __init__(self):
        pass

    @classmethod
    async def create_instance(cls, account: str, password: str, executors=None, max_workers=1, tz='Asia/Chongqing'):
        global _executors

        _instance = Fetcher()
        jq.auth(account, password)

        # noinspection PyProtectedMember
        if executors is None or executors._max_workers > max_workers:
            _executors = ThreadPoolExecutor(max_workers=min(1, max_workers))
        else:
            _executors = executors

        return _instance

    @async_concurrent(_executors)
    def get_bars(self, sec: str, end_at: Arrow, n_bars: int, frame_type: FrameType) -> np.array:
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
        end_at = tf.shift(end_at, 1, frame_type)
        if isinstance(end_at, Arrow):
            end_at = end_at.datetime

        try:
            logger.info("fetching %s n_bars for %s end_at %s", n_bars, sec, end_at)
            data = jq.get_bars(sec, n_bars, unit=frame_type.value, end_dt=end_at, fq_ref_date=None, df=False,
                               fields=['date', 'open', 'high', 'low', 'close', 'volume', 'money', 'factor'],
                               include_now=True)
            data.dtype.names = ['frame', 'open', 'high', 'low', 'close', 'volume', 'amount', 'factor']
            return data
        except Exception as e:
            logger.exception(e)
            if str(e).find("最大查询限制") != -1:
                raise FetcherQuotaError("Exceeded JQDataSDK Quota")

    @async_concurrent(_executors)
    def get_security_list(self) -> np.ndarray:
        types = ['stock', 'fund', 'index', 'futures', 'etf', 'lof']
        securities = jq.get_all_securities(types)
        securities.insert(0, 'code', securities.index)
        return securities.values

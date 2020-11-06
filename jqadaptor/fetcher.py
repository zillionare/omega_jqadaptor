"""joinquant adaptor for zillionare"""

__author__ = """Aaron Yang"""
__email__ = "code@jieyu.ai"
__version__ = "0.1.1"

# -*- coding: utf-8 -*-
import datetime
import logging
from typing import List, Union

import pandas as pd
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

    warnings.warn("请安装jqdatasdk 1.8以上版本。程序即将退出。")
    import sys

    sys.exit(-1)
import numpy as np

logger = logging.getLogger(__file__)


@singleton
class Fetcher:
    """
    JQFetcher is a subclass of QuotesFetcher
    """

    tz = pytz.timezone("Asia/Shanghai")

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
        cls.tz = tz.gettz(kwargs.get("tz", "Asia/Shanghai"))
        _instance = Fetcher()
        account = str(kwargs.get("account"))
        password = str(kwargs.get("password"))

        jq.auth(str(account), str(password))
        logger.info("jqdata sdk login success")

        return _instance

    async def get_bars_batch(
        self,
        secs: List[str],
        end_at: datetime.datetime,
        n_bars: int,
        frame_type: FrameType,
        include_unclosed=True,
    ) -> np.array:
        if type(end_at) not in [datetime.date, datetime.datetime]:
            raise TypeError(
                "end_at must by type of datetime.date or datetime.datetime")

        # has to use type rather than isinstance, since the latter always return true
        # when check if isinstance(datetime.datetime, datetime.date)
        if type(end_at) is datetime.date:  # pylint: disable=unidiomatic-typecheck
            end_at = datetime.datetime(end_at.year, end_at.month, end_at.day,
                                       15)
        resp = jq.get_bars(
            secs,
            n_bars,
            frame_type.value,
            fields=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "money",
                "factor",
            ],
            end_dt=end_at,
            include_now=include_unclosed,
            fq_ref_date=end_at,
            df=False,
        )
        results = {}
        for code, bars in resp.items():
            bars = np.array(
                bars,
                dtype=[
                    ("frame", "O"),
                    ("open", "f4"),
                    ("high", "f4"),
                    ("low", "f4"),
                    ("close", "f4"),
                    ("volume", "f8"),
                    ("amount", "f8"),
                    ("factor", "f4"),
                ],
            )

            if frame_type in tf.minute_level_frames:
                bars["frame"] = [
                    frame.astimezone(self.tz) for frame in bars["frame"]
                ]

            results[code] = bars

        return results

    async def get_bars(
        self,
        sec: str,
        end_at: Union[datetime.date, datetime.datetime],
        n_bars: int,
        frame_type: FrameType,
        include_unclosed=True,
    ) -> np.array:
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
        logger.debug("fetching %s bars for %s until %s", n_bars, sec, end_at)

        if type(end_at) not in [datetime.date, datetime.datetime]:
            raise TypeError(
                "end_at must by type of datetime.date or datetime.datetime")

        if type(end_at) is datetime.date:  # noqa
            end_at = datetime.datetime(end_at.year, end_at.month, end_at.day,
                                       15)
        try:
            bars = jq.get_bars(
                sec,
                n_bars,
                unit=frame_type.value,
                end_dt=end_at,
                fq_ref_date=None,
                df=False,
                fields=[
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "money",
                    "factor",
                ],
                include_now=include_unclosed,
            )
            # convert to omega supported format
            bars = np.array(
                bars,
                dtype=[
                    ("frame", "O"),
                    ("open", "f4"),
                    ("high", "f4"),
                    ("low", "f4"),
                    ("close", "f4"),
                    ("volume", "f8"),
                    ("amount", "f8"),
                    ("factor", "f4"),
                ],
            )
            if len(bars) == 0:
                logger.warning("fetching %s(%s,%s) returns empty result", sec,
                               n_bars, end_at)
                return bars

            if frame_type in tf.minute_level_frames:
                bars["frame"] = [
                    frame.astimezone(self.tz) for frame in bars["frame"]
                ]

            return bars
        except Exception as e:  # pylint: disable=broad-except
            logger.exception(e)
            if str(e).find("最大查询限制") != -1:
                raise FetcherQuotaError("Exceeded JQDataSDK Quota") from e
            else:
                raise e

    async def get_security_list(self) -> np.ndarray:
        """

        Returns:

        """
        types = ["stock", "fund", "index", "futures", "etf", "lof"]
        securities = jq.get_all_securities(types)
        securities.insert(0, "code", securities.index)

        # remove client dependency of pandas
        securities["start_date"] = securities["start_date"].apply(
            lambda s: f"{s.year:04}-{s.month:02}-{s.day:02}")
        securities["end_date"] = securities["end_date"].apply(
            lambda s: f"{s.year:04}-{s.month:02}-{s.day:02}")
        return securities.values

    async def get_all_trade_days(self) -> np.array:
        return jq.get_all_trade_days()

    async def get_valuation(self, codes: Union[str, List[str]],
                            day: datetime.date) -> np.ndarray:
        if isinstance(codes, str):
            codes = [codes]

        df = jq.get_fundamentals(
            jq.query(jq.valuation).filter(
                # 这里不能使用 in 操作, 要使用in_()函数
                jq.valuation.code.in_(codes)),
            date=str(day),
        )

        return self._to_numpy(df)

    def _to_numpy(self, df: pd.DataFrame) -> np.array:
        df["date"] = pd.to_datetime(df["day"]).dt.date

        # translate joinquant definition to zillionare definition
        fields = {
            "code": "code",
            "pe_ratio": "pe",
            "turnover_ratio": "turnover",
            "pb_ratio": "pb",
            "ps_ratio": "ps",
            "pcf_ratio": "pcf",
            "capitalization": "capital",
            "market_cap": "market_cap",
            "circulating_cap": "circulating_cap",
            "circulating_market_cap": "circulating_market_cap",
            "pe_ratio_lyr": "pe_lyr",
            "date": "date",
        }

        df = df[fields.keys()]

        dtypes = [(fields[_name], _type)
                  for _name, _type in zip(df.dtypes.index, df.dtypes)]

        # the following line will return a np.recarray, which is slightly slow than
        # structured array, so it's commented out
        # return np.rec.fromrecords(valuation.values, names=valuation.columns.tolist())
        # to get a structued array
        return np.array([tuple(x) for x in df.to_numpy()], dtype=dtypes)

    async def get_valuation_in_range(self, code: str, day: datetime.date,
                                     n: int) -> np.array:
        """get `n` of `code`'s valuation records, end at day.

        Args:
            code (str): [description]
            day (datetime.date): [description]
            n (int): [description]

        Returns:
            np.array: [description]
        """
        q = jq.query(jq.valuation).filter(jq.valuation.code == code)

        df = jq.get_fundamentals_continuously(q,
                                              count=n,
                                              end_date=day,
                                              panel=False)

        return self._to_numpy(df)

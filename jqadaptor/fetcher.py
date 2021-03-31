"""joinquant adaptor for zillionare"""

__author__ = """Aaron Yang"""
__email__ = "code@jieyu.ai"
__version__ = "0.1.1"

# -*- coding: utf-8 -*-
import datetime
import functools
import logging
from typing import List, Union

import dateutil
import jqdatasdk as jq
import numpy as np
import pandas as pd
import pytz

logger = logging.getLogger(__name__)


minute_level_frames = ["60m", "30m", "15m", "5m", "1m"]


class FetcherQuotaError(BaseException):
    """quotes fetcher quota exceed"""

    pass


def singleton(cls):
    """Make a class a Singleton class

    Examples:
        >>> @singleton
        ... class Foo:
        ...     # this is a singleton class
        ...     pass

    """
    instances = {}

    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class Fetcher:
    """
    JQFetcher is a subclass of QuotesFetcher
    """

    connected = False
    tz = pytz.timezone("Asia/Shanghai")

    def __init__(self):
        pass

    @classmethod
    async def create_instance(
        cls, account: str, password: str, tz: str = "Asia/Shanghai", **kwargs
    ):
        """
        创建jq_adaptor实例。 kwargs用来接受多余但不需要的参数。
        Args:
            account: str
            password: str
            tz: str
            kwargs: not required
        Returns:

        """
        cls.tz = dateutil.tz.gettz(tz)
        _instance = Fetcher()

        account = str(account)
        password = str(password)

        try:
            jq.auth(account, password)

            cls.connected = True
            logger.info("jqdata sdk login success")
        except Exception as e:
            cls.connected = False
            logger.exception(e)
            logger.warning(
                "login failed, account %s, password: %s",
                account[: min(4, len(account))].ljust(7, "*"),
                password[:2],
            )

        return _instance

    async def get_bars_batch(
        self,
        secs: List[str],
        end_at: datetime.datetime,
        n_bars: int,
        frame_type: str,
        include_unclosed=True,
    ) -> np.array:
        if not self.connected:
            logger.warning("not connected.")
            return None

        if type(end_at) not in [datetime.date, datetime.datetime]:
            raise TypeError("end_at must by type of datetime.date or datetime.datetime")

        # has to use type rather than isinstance, since the latter always return true
        # when check if isinstance(datetime.datetime, datetime.date)
        if type(end_at) is datetime.date:  # pylint: disable=unidiomatic-typecheck
            end_at = datetime.datetime(end_at.year, end_at.month, end_at.day, 15)
        resp = jq.get_bars(
            secs,
            n_bars,
            frame_type,
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

            if frame_type in minute_level_frames:
                bars["frame"] = [
                    frame.replace(tzinfo=self.tz) for frame in bars["frame"]
                ]

            results[code] = bars

        return results

    async def get_bars(
        self,
        sec: str,
        end_at: Union[datetime.date, datetime.datetime],
        n_bars: int,
        frame_type: str,
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
        if not self.connected:
            logger.warning("not connected")
            return None

        logger.debug("fetching %s bars for %s until %s", n_bars, sec, end_at)

        if type(end_at) not in [datetime.date, datetime.datetime]:
            raise TypeError("end_at must by type of datetime.date or datetime.datetime")

        if type(end_at) is datetime.date:  # noqa
            end_at = datetime.datetime(end_at.year, end_at.month, end_at.day, 15)
        try:
            bars = jq.get_bars(
                sec,
                n_bars,
                unit=frame_type,
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
                logger.warning(
                    "fetching %s(%s,%s) returns empty result", sec, n_bars, end_at
                )
                return bars

            if frame_type in minute_level_frames:
                bars["frame"] = [
                    frame.replace(tzinfo=self.tz) for frame in bars["frame"]
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
        if not self.connected:
            logger.warning("not connected")
            return None

        types = ["stock", "fund", "index", "futures", "etf", "lof"]
        securities = jq.get_all_securities(types)
        securities.insert(0, "code", securities.index)

        # remove client dependency of pandas
        securities["start_date"] = securities["start_date"].apply(
            lambda s: f"{s.year:04}-{s.month:02}-{s.day:02}"
        )
        securities["end_date"] = securities["end_date"].apply(
            lambda s: f"{s.year:04}-{s.month:02}-{s.day:02}"
        )
        return securities.values

    async def get_all_trade_days(self) -> np.array:
        if not self.connected:
            logger.warning("not connected")
            return None

        return jq.get_all_trade_days()

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
            "date": "frame",
        }

        df = df[fields.keys()]

        dtypes = [
            (fields[_name], _type) for _name, _type in zip(df.dtypes.index, df.dtypes)
        ]

        # the following line will return a np.recarray, which is slightly slow than
        # structured array, so it's commented out
        # return np.rec.fromrecords(valuation.values, names=valuation.columns.tolist())
        # to get a structued array
        return np.array([tuple(x) for x in df.to_numpy()], dtype=dtypes)

    async def get_valuation(
        self, codes: Union[str, List[str]], day: datetime.date, n: int = 1
    ) -> np.array:
        if not self.connected:
            logger.warning("not connected")
            return None

        """get `n` of `code`'s valuation records, end at day.

        对同一证券，返回的数据按升序排列（但取决于上游数据源）
        Args:
            code (str): [description]
            day (datetime.date): [description]
            n (int): [description]

        Returns:
            np.array: [description]
        """
        if isinstance(codes, str):
            codes = [codes]

        if codes is None:
            q = jq.query(jq.valuation)
        else:
            q = jq.query(jq.valuation).filter(jq.valuation.code.in_(codes))

        records = jq.get_fundamentals_continuously(
            q, count=n, end_date=day, panel=False
        )

        return self._to_numpy(records)

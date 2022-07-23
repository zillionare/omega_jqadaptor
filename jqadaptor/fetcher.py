"""joinquant adaptor for zillionare"""

__author__ = """Aaron Yang"""
__email__ = "code@jieyu.ai"
__version__ = "0.1.1"

# -*- coding: utf-8 -*-
import asyncio
import copy
import datetime
import functools
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Union

import jqdatasdk as jq
import numpy as np
import pandas as pd
import pytz
from coretypes import QuotesFetcher, bars_dtype
from numpy.typing import ArrayLike
from pandas.core.frame import DataFrame
from sqlalchemy import func

logger = logging.getLogger(__name__)

minute_level_frames = ["60m", "30m", "15m", "5m", "1m"]


def async_concurrent(executors):
    def decorator(f):
        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            p = functools.partial(f, *args, **kwargs)
            loop = asyncio.get_running_loop()
            try:
                return await loop.run_in_executor(executors, p)
            except Exception as e:  # pylint: disable=broad-except
                logger.exception(e)
                if str(e).find("最大查询限制") != -1:
                    raise FetcherQuotaError("Exceeded JQDataSDK Quota") from e
                elif str(e).find("账号过期") != -1:
                    logger.warning(
                        "account %s expired, please contact jqdata", Fetcher.account
                    )
                    raise AccountExpiredError(
                        f"Account {Fetcher.account} expired"
                    ) from e
                else:
                    raise e

        return wrapper

    return decorator


class FetcherQuotaError(BaseException):
    """quotes fetcher quota exceed"""

    pass


class AccountExpiredError(BaseException):
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
class Fetcher(QuotesFetcher):
    """
    JQFetcher is a subclass of QuotesFetcher
    """

    connected = False
    tz = pytz.timezone("Asia/Shanghai")
    executor = ThreadPoolExecutor(1)

    account = None
    password = None

    def __init__(self):
        pass

    @classmethod
    @async_concurrent(executor)
    def create_instance(cls, account: str, password: str, **kwargs):
        """创建jq_adaptor实例。 kwargs用来接受多余但不需要的参数。

        Args:
            account: 聚宽账号
            password: 聚宽密码
            kwargs: not required
        Returns:
            None
        """

        cls.login(account, password, **kwargs)

    @async_concurrent(executor)
    def get_bars_batch(
        self,
        secs: List[str],
        end_at: datetime.datetime,
        n_bars: int,
        frame_type: str,
        include_unclosed=True,
    ) -> Dict[str, np.ndarray]:
        """批量获取多只股票的行情数据

        Args:
            secs: 股票代码列表
            end_at: 查询的结束时间
            n_bars: 查询的记录数
            frame_type: 查询的周期，比如1m, 5m, 15m, 30m, 60m, 1d, 1w, 1M, 1Q, 1Y等
            include_unclosed: 如果`end_at`没有指向`frame_type`的收盘时间，是否只取截止到上一个已收盘的数据。
        Returns:
            字典，其中key为股票代码，值为对应的行情数据，类型为bars_dtype.
        """
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

        return {code: bars.astype(bars_dtype) for code, bars in resp.items()}

    @async_concurrent(executor)
    def get_bars(
        self,
        sec: str,
        end_at: Union[datetime.date, datetime.datetime],
        n_bars: int,
        frame_type: str,
        include_unclosed=True,
    ) -> np.ndarray:
        """获取`sec`在`end_at`时刻的`n_bars`个`frame_type`的行情数据

        Args:
            sec: 股票代码
            end_at: 查询的结束时间
            n_bars: 查询的记录数
            frame_type: 查询的周期，比如1m, 5m, 15m, 30m, 60m, 1d, 1w, 1M, 1Q, 1Y等
            include_unclosed: 如果`end_at`没有指向`frame_type`的收盘时间，是否只取截止到上一个已收盘的数据。
        Returns:
            行情数据，类型为bars_dtype.
        """
        if not self.connected:
            logger.warning("not connected")
            return None

        logger.debug("fetching %s bars for %s until %s", n_bars, sec, end_at)

        if type(end_at) not in [datetime.date, datetime.datetime]:
            raise TypeError("end_at must by type of datetime.date or datetime.datetime")

        if type(end_at) is datetime.date:  # noqa
            end_at = datetime.datetime(end_at.year, end_at.month, end_at.day, 15)

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
        bars = bars.astype(bars_dtype)
        if len(bars) == 0:
            logger.warning(
                "fetching %s(%s,%s) returns empty result", sec, n_bars, end_at
            )
            return bars

        return bars

    @async_concurrent(executor)
    def get_price(
        self,
        sec: Union[List, str],
        end_date: Union[str, datetime.datetime],
        n_bars: int,
        frame_type: str,
    ) -> Dict[str, np.ndarray]:
        """获取一支或者多支股票的价格数据
            一般我们使用`get_bars`来获取股票的行情数据。这个方法用以数据校验。
        Args:
            sec: 股票代码或者股票代码列表
            end_date: 查询的结束时间
            n_bars: 查询的记录数
            frame_type: 查询的周期，比如1m, 5m, 15m, 30m, 60m, 1d, 1w, 1M, 1Q, 1Y等
        Returns:
            字典，其中key为股票代码，值为对应的行情数据，类型为bars_dtype.
        """
        if type(end_date) not in (str, datetime.date, datetime.datetime):
            raise TypeError(
                "end_at must by type of datetime.date or datetime.datetime or str"
            )
        if type(sec) not in (list, str):
            raise TypeError("sec must by type of list or str")
        fields = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "money",
            "factor",
        ]
        params = {
            "security": sec,
            "end_date": end_date,
            "fields": fields,
            "fq": None,
            "fill_paused": False,
            "frequency": frame_type,
            "count": n_bars,
            "skip_paused": True,
        }
        df = jq.get_price(**params)
        # 处理时间 转换成datetime
        temp_bars_dtype = copy.deepcopy(bars_dtype)
        temp_bars_dtype.insert(1, ("code", "O"))
        ret = {}
        for code, group in df.groupby("code"):
            df = group[
                [
                    "time",  # python object either of Frame type
                    "code",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "money",
                    "factor",
                ]
            ].sort_values("time")
            bars = df.to_records(index=False).astype(temp_bars_dtype)
            bars["frame"] = [x.to_pydatetime() for x in df["time"]]
            ret[code] = bars.view(np.ndarray)

        return ret

    @async_concurrent(executor)
    def get_finance_xrxd_info(
        self, dt_start: datetime.date, dt_end: datetime.date
    ) -> list:
        """上市公司分红送股（除权除息）数据 / 2005至今，8:00更新

        聚宽提供的数据是按季组织的。这里的`dt_start`和`dt_end`是指实际季报/年报的时间，而不是实际除权除息的时间。

        Args:
            dt_start: 开始日期
            dt_end: 结束日期
        Returns:
            分红送股数据,其每一个元素是一个元组，形如：('002589.XSHE', datetime.date(2022, 7, 22), '10派0.09元(含税)', 0.09, 0.0, 0.0, 0.09, datetime.date(2021, 12, 31), '实施方案', '10派0.09元(含税)', datetime.date(2099, 1, 1))
        """
        if not self.connected:
            logger.warning("not connected")
            return None

        # dt_end一般为当天，dt_start一般为dt_end-366天
        if dt_start is None or dt_end is None:
            return None

        q_for_count = jq.query(func.count(jq.finance.STK_XR_XD.id))
        q_for_count = q_for_count.filter(
            jq.finance.STK_XR_XD.a_xr_date.isnot(None),
            jq.finance.STK_XR_XD.report_date >= dt_start,
            jq.finance.STK_XR_XD.report_date <= dt_end,
        )

        q = jq.query(jq.finance.STK_XR_XD).filter(
            jq.finance.STK_XR_XD.a_xr_date.isnot(None),
            jq.finance.STK_XR_XD.report_date >= dt_start,
            jq.finance.STK_XR_XD.report_date <= dt_end,
        )

        reports_count = jq.finance.run_query(q_for_count)["count_1"][0]

        page = 0
        dfs: List[pd.DataFrame] = []
        while page * 3000 < reports_count:
            df1 = jq.finance.run_query(q.offset(page * 3000).limit(3000))
            dfs.append(df1)
            page += 1
        if len(dfs) == 0:
            return None
        df = pd.concat(dfs)

        reports = []
        for _, row in df.iterrows():
            a_xr_date = row["a_xr_date"]
            if a_xr_date is None:  # 还未确定的方案不登记
                continue

            code = row["code"]
            # company_name = row['company_name']  # 暂时不存公司名字，没实际意义
            report_date = row["report_date"]
            board_plan_bonusnote = row["board_plan_bonusnote"]
            implementation_bonusnote = row["implementation_bonusnote"]  # 有实施才有公告

            bonus_cancel_pub_date = row["bonus_cancel_pub_date"]
            if bonus_cancel_pub_date is None:  # 如果不是2099.1.1，即发生了取消事件
                bonus_cancel_pub_date = datetime.date(2099, 1, 1)

            bonus_ratio_rmb = row["bonus_ratio_rmb"]
            if bonus_ratio_rmb is None or math.isnan(bonus_ratio_rmb):
                bonus_ratio_rmb = 0.0
            dividend_ratio = row["dividend_ratio"]
            if dividend_ratio is None or math.isnan(dividend_ratio):
                dividend_ratio = 0.0
            transfer_ratio = row["transfer_ratio"]
            if transfer_ratio is None or math.isnan(transfer_ratio):
                transfer_ratio = 0.0
            at_bonus_ratio_rmb = row["at_bonus_ratio_rmb"]
            if at_bonus_ratio_rmb is None or math.isnan(at_bonus_ratio_rmb):
                at_bonus_ratio_rmb = 0.0
            plan_progress = row["plan_progress"]

            record = (
                code,
                a_xr_date,
                board_plan_bonusnote,
                bonus_ratio_rmb,
                dividend_ratio,
                transfer_ratio,
                at_bonus_ratio_rmb,
                report_date,
                plan_progress,
                implementation_bonusnote,
                bonus_cancel_pub_date,
            )
            reports.append(record)

        return reports

    @async_concurrent(executor)
    def get_security_list(self, date: datetime.date = None) -> np.ndarray:
        """获取`date`日的证券列表

        Args:
            date: 日期。如果为None，则取当前日期的证券列表
        Returns:
            证券列表, dtype为[('code', 'O'), ('display_name', 'O'), ('name', 'O'), ('start_date', 'O'), ('end_date', 'O'), ('type', 'O')]的structured array
        """
        if not self.connected:
            logger.warning("not connected")
            return None

        types = ["stock", "fund", "index", "etf", "lof"]
        securities = jq.get_all_securities(types, date)
        securities.insert(0, "code", securities.index)

        # remove client dependency of pandas
        securities["start_date"] = securities["start_date"].apply(
            lambda s: f"{s.year:04}-{s.month:02}-{s.day:02}"
        )
        securities["end_date"] = securities["end_date"].apply(
            lambda s: f"{s.year:04}-{s.month:02}-{s.day:02}"
        )
        return securities.to_records(index=False)

    @async_concurrent(executor)
    def get_all_trade_days(self) -> np.ndarray:
        """获取所有交易日的日历

        Returns:
            交易日日历, dtype为datetime.date的numpy array
        """
        if not self.connected:
            logger.warning("not connected")
            return None

        return jq.get_all_trade_days()

    def _to_numpy(self, df: pd.DataFrame) -> np.ndarray:
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

    @async_concurrent(executor)
    def get_valuation(
        self, codes: Union[str, List[str]], day: datetime.date, n: int = 1
    ) -> np.ndarray:
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

    @staticmethod
    def __dataframe_to_structured_array(
        df: pd.DataFrame, dtypes: List[Tuple] = None
    ) -> ArrayLike:
        """convert dataframe (with all columns, and index possibly) to numpy structured arrays

        `len(dtypes)` should be either equal to `len(df.columns)` or `len(df.columns) + 1`. In the later case, it implies to include `df.index` into converted array.

        Args:
            df: the one needs to be converted
            dtypes: Defaults to None. If it's `None`, then dtypes of `df` is used, in such case, the `index` of `df` will not be converted.

        Returns:
            ArrayLike: [description]
        """
        v = df
        if dtypes is not None:
            dtypes_in_dict = {key: value for key, value in dtypes}

            col_len = len(df.columns)
            if len(dtypes) == col_len + 1:
                v = df.reset_index()

                rename_index_to = set(dtypes_in_dict.keys()).difference(set(df.columns))
                v.rename(columns={"index": list(rename_index_to)[0]}, inplace=True)
            elif col_len != len(dtypes):
                raise ValueError(
                    f"length of dtypes should be either {col_len} or {col_len + 1}, is {len(dtypes)}"
                )

            # re-arrange order of dtypes, in order to align with df.columns
            dtypes = []
            for name in v.columns:
                dtypes.append((name, dtypes_in_dict[name]))
        else:
            dtypes = df.dtypes

        return np.array(np.rec.fromrecords(v.values), dtype=dtypes)

    @async_concurrent(executor)
    def get_trade_price_limits(
        self, sec: Union[List, str], dt: Union[str, datetime.datetime, datetime.date]
    ) -> np.ndarray:
        """获取某个时间点的交易价格限制，即涨停价和跌停价

        Returns:
            an numpy structured array which dtype is:
            [('frame', 'O'), ('code', 'O'), ('high_limit', '<f4'), ('low_limit', '<f4')]

            the frame is python datetime.date object
        """
        if type(dt) not in (str, datetime.date, datetime.datetime):
            raise TypeError(
                "end_at must by type of datetime.date or datetime.datetime or str"
            )
        if type(sec) not in (list, str):
            raise TypeError("sec must by type of list or str")

        fields = ["high_limit", "low_limit"]
        params = {
            "security": sec,
            "end_date": dt,
            "fields": fields,
            "fq": None,
            "fill_paused": False,
            "frequency": "1d",
            "count": 1,
            "skip_paused": True,
        }
        df = jq.get_price(**params)

        dtype = [
            ("frame", "O"),
            ("code", "O"),
            ("high_limit", "<f4"),
            ("low_limit", "<f4"),
        ]
        if len(df) == 0:
            return None
        bars = df.to_records(index=False).astype(dtype)
        bars["frame"] = df["time"].apply(lambda x: x.to_pydatetime().date())
        return bars

    def _to_fund_numpy(self, df: pd.DataFrame) -> np.array:
        df["start_date"] = pd.to_datetime(df["start_date"]).dt.date
        df["end_date"] = pd.to_datetime(df["end_date"]).dt.date

        fields = {
            "main_code": "code",
            "name": "name",
            "advisor": "advisor",
            "trustee": "trustee",
            "operate_mode_id": "operate_mode_id",
            "operate_mode": "operate_mode",
            "start_date": "start_date",
            "end_date": "end_date",
            "underlying_asset_type_id": "underlying_asset_type_id",
            "underlying_asset_type": "underlying_asset_type",
        }

        df = df[fields.keys()]

        dtypes = [
            (fields[_name], _type) for _name, _type in zip(df.dtypes.index, df.dtypes)
        ]
        return np.array([tuple(x) for x in df.to_numpy()], dtype=dtypes)

    @async_concurrent(executor)
    def get_fund_list(self, codes: Union[str, List[str]] = None) -> np.ndarray:
        """获取所有的基金基本信息
        Args:
            codes: 可以是一个基金代码，或者是一个列表，如果为空，则获取所有的基金
        Returns:
            np.array: [基金的基本信息]
        """
        if not self.connected:
            logger.warning("not connected")
            return None

        if codes and isinstance(codes, str):
            codes = [codes]
        fund_count_q = jq.query(func.count(jq.finance.FUND_MAIN_INFO.id))
        fund_q = jq.query(jq.finance.FUND_MAIN_INFO).order_by(
            jq.finance.FUND_MAIN_INFO.id.asc()
        )
        if codes:
            fund_count_q = fund_count_q.filter(
                jq.finance.FUND_MAIN_INFO.main_code.in_(codes)
            )
            fund_q = fund_q.filter(jq.finance.FUND_MAIN_INFO.main_code.in_(codes))

        fund_count = jq.finance.run_query(fund_count_q)["count_1"][0]
        dfs: List[pd.DataFrame] = []
        page = 0
        while page * 3000 < fund_count:
            df1 = jq.finance.run_query(fund_q.offset(page * 3000).limit(3000))
            dfs.append(df1)
            page += 1
        funds: DataFrame = (
            pd.concat(dfs)
            if dfs
            else pd.DataFrame(
                columns=[
                    "main_code",
                    "name",
                    "advisor",
                    "trustee",
                    "operate_mode_id",
                    "operate_mode",
                    "start_date",
                    "end_date",
                    "underlying_asset_type_id",
                    "underlying_asset_type",
                ]
            )
        )
        funds["start_date"] = funds["start_date"].apply(
            lambda s: f"{s.year:04}-{s.month:02}-{s.day:02}" if s else "2099-01-01"
        )
        funds["end_date"] = funds["end_date"].apply(
            lambda s: f"{s.year:04}-{s.month:02}-{s.day:02}" if s else "2099-01-01"
        )

        return self._to_fund_numpy(funds)

    def _to_fund_portfolio_stock_numpy(self, df: pd.DataFrame) -> np.array:
        fields = {
            "code": "code",
            "period_start": "period_start",
            "period_end": "period_end",
            "pub_date": "pub_date",
            "report_type_id": "report_type_id",
            "report_type": "report_type",
            "rank": "rank",
            "symbol": "symbol",
            "name": "name",
            "shares": "shares",
            "market_cap": "market_cap",
            "proportion": "proportion",
            "deadline": "deadline",
        }

        df = df[fields.keys()]

        dtypes = [
            (fields[_name], _type) for _name, _type in zip(df.dtypes.index, df.dtypes)
        ]
        return np.array([tuple(x) for x in df.to_numpy()], dtype=dtypes)

    @async_concurrent(executor)
    def get_fund_portfolio_stock(
        self, codes: Union[str, List[str]], pub_date: Union[str, datetime.date] = None
    ) -> np.array:
        if not self.connected:
            logger.warning("not connected")
            return None
        if codes and isinstance(codes, str):
            codes = [codes]
        fund_count_q = jq.query(func.count(jq.finance.FUND_PORTFOLIO_STOCK.id))
        q = jq.query(jq.finance.FUND_PORTFOLIO_STOCK)
        if codes:
            q = q.filter(jq.finance.FUND_PORTFOLIO_STOCK.code.in_(codes))
            fund_count_q = fund_count_q.filter(
                jq.finance.FUND_PORTFOLIO_STOCK.code.in_(codes)
            )

        if pub_date:
            q = q.filter(jq.finance.FUND_PORTFOLIO_STOCK.pub_date == pub_date)
            fund_count_q = fund_count_q.filter(
                jq.finance.FUND_PORTFOLIO_STOCK.pub_date == pub_date
            )

        fund_count = jq.finance.run_query(fund_count_q)["count_1"][0]

        dfs: List[pd.DataFrame] = []
        page = 0
        while page * 3000 < fund_count:
            df1 = jq.finance.run_query(q.offset(page * 3000).limit(3000))
            dfs.append(df1)
            page += 1
        df: DataFrame = (
            pd.concat(dfs)
            if dfs
            else pd.DataFrame(
                columns=[
                    "code",
                    "period_start",
                    "period_end",
                    "pub_date",
                    "report_type_id",
                    "report_type",
                    "rank",
                    "symbol",
                    "name",
                    "shares",
                    "market_cap",
                    "proportion",
                    "deadline",
                ]
            )
        )
        df["deadline"] = df["pub_date"].map(
            lambda x: (
                x
                + pd.tseries.offsets.DateOffset(
                    months=-((x.month - 1) % 3), days=1 - x.day
                )
                - datetime.timedelta(days=1)
            ).date()
        )
        df = df.sort_values(
            by=["code", "pub_date", "symbol", "report_type", "period_end"],
            ascending=[False, False, False, False, False],
        ).drop_duplicates(
            subset=[
                "code",
                "pub_date",
                "symbol",
                "report_type",
            ],
            keep="first",
        )
        if df.empty:
            df = pd.DataFrame(
                columns=[
                    "code",
                    "period_start",
                    "period_end",
                    "pub_date",
                    "report_type_id",
                    "report_type",
                    "rank",
                    "symbol",
                    "name",
                    "shares",
                    "market_cap",
                    "proportion",
                    "deadline",
                ]
            )
        else:
            df = df.groupby(by="code").apply(lambda x: x.nlargest(10, "shares"))
        return self._to_fund_portfolio_stock_numpy(df)

    def _to_fund_net_value_numpy(self, df: pd.DataFrame) -> np.ndarray:
        df["day"] = pd.to_datetime(df["day"]).dt.date

        fields = {
            "code": "code",
            "net_value": "net_value",
            "sum_value": "sum_value",
            "factor": "factor",
            "acc_factor": "acc_factor",
            "refactor_net_value": "refactor_net_value",
            "day": "day",
        }

        df = df[fields.keys()]

        dtypes = [
            (fields[_name], _type) for _name, _type in zip(df.dtypes.index, df.dtypes)
        ]
        return np.array([tuple(x) for x in df.to_numpy()], dtype=dtypes)

    @async_concurrent(executor)
    def get_fund_net_value(
        self,
        codes: Union[str, List[str]],
        day: datetime.date = None,
    ) -> np.ndarray:
        if not self.connected:
            logger.warning("not connected")
            return None
        if codes and isinstance(codes, str):
            codes = [codes]

        day = day or (datetime.datetime.now().date() - datetime.timedelta(days=1))
        q = jq.query(jq.finance.FUND_NET_VALUE).filter(
            jq.finance.FUND_NET_VALUE.day == day
        )
        q_count = jq.query(func.count(jq.finance.FUND_NET_VALUE.id)).filter(
            jq.finance.FUND_NET_VALUE.day == day
        )
        if codes:
            q = q.filter(jq.finance.FUND_NET_VALUE.code.in_(codes))
            q_count = q_count.filter(jq.finance.FUND_NET_VALUE.code.in_(codes))
        fund_count = jq.finance.run_query(q_count)["count_1"][0]
        dfs: List[pd.DataFrame] = []
        page = 0
        while page * 3000 < fund_count:
            df1: DataFrame = jq.finance.run_query(q.offset(page * 3000).limit(3000))
            if not df1.empty:
                dfs.append(df1)
                page += 1
        df = (
            pd.concat(dfs)
            if dfs
            else pd.DataFrame(
                columns=[
                    "code",
                    "net_value",
                    "sum_value",
                    "factor",
                    "acc_factor",
                    "refactor_net_value",
                    "day",
                ]
            )
        )
        return self._to_fund_net_value_numpy(df)

    def _to_fund_share_daily_numpy(self, df: pd.DataFrame) -> np.ndarray:
        df["day"] = pd.to_datetime(df["pub_date"]).dt.date

        fields = {
            "code": "code",
            "total_tna": "total_tna",
            "day": "date",
            "name": "name",
        }

        df = df[fields.keys()]

        dtypes = [
            (fields[_name], _type) for _name, _type in zip(df.dtypes.index, df.dtypes)
        ]
        return np.array([tuple(x) for x in df.to_numpy()], dtype=dtypes)

    @async_concurrent(executor)
    def get_fund_share_daily(
        self, codes: Union[str, List[str]] = None, day: datetime.date = None
    ) -> np.ndarray:
        if not self.connected:
            logger.warning("not connected")
            return None

        if codes and isinstance(codes, str):
            codes = [codes]
        day = day or (datetime.datetime.now().date() - datetime.timedelta(days=1))

        q_fund_fin_indicator = jq.query(jq.finance.FUND_FIN_INDICATOR).filter(
            jq.finance.FUND_FIN_INDICATOR.pub_date == day
        )
        if codes:
            q_fund_fin_indicator = q_fund_fin_indicator.filter(
                jq.finance.FUND_FIN_INDICATOR.code.in_(codes)
            )
        df: DataFrame = jq.finance.run_query(q_fund_fin_indicator)
        df = df.drop_duplicates(subset=["code", "pub_date"], keep="first")
        df["total_tna"] = df["total_tna"].fillna(0)
        return self._to_fund_share_daily_numpy(df)

    @async_concurrent(executor)
    def get_quota(self) -> Dict[str, int]:
        """查询quota使用情况

        返回值为一个dict, key为"total"，"spare"
        Returns:
            dict: quota
        """
        quota = jq.get_query_count()
        assert "total" in quota
        assert "spare" in quota

        return quota

    @classmethod
    def login(cls, account, password, **kwargs):
        """登录"""
        account = str(account)
        password = str(password)

        logger.info(
            "login jqdatasdk with account %s, password: %s",
            account[: min(4, len(account))].ljust(7, "*"),
            password[:2],
        )
        try:
            jq.auth(account, password)
            cls.connected = True
            cls.account = account
            cls.password = password
            logger.info("jqdatasdk login success")
        except Exception as e:
            cls.connected = False
            logger.exception(e)
            logger.warning("jqdatasdk login failed")

    @classmethod
    def logout(cls):
        """退出登录"""
        return jq.logout()

    @classmethod
    def reconnect(cls):
        cls.logout()
        cls.login(cls.account, cls.password)

    @classmethod
    def result_size_limit(cls, op) -> int:
        """单次查询允许返回的最大记录数"""
        return {}.get(op, 3000)

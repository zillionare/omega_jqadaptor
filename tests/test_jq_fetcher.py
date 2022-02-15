#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import unittest
from unittest import mock
import numpy as np

import arrow

import jqadaptor
from jqadaptor.fetcher import FetcherQuotaError

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class TestJQ(unittest.IsolatedAsyncioTestCase):
    fetcher: "QuotesFetcher"  # type: ignore # noqa

    async def asyncSetUp(self) -> None:  # pylint: disable=invalid-overridden-method
        try:
            account = os.environ["JQ_ACCOUNT"]
            password = os.environ["JQ_PASSWORD"]
            blablah = "blahblah"

            self.fetcher = await jqadaptor.create_instance(
                account=account, password=password, blablah=blablah
            )

        except Exception as e:
            logger.exception(e)

    async def test_failed_login(self):
        fetcher = await jqadaptor.create_instance(account="wrong", password="wrong")

        log = logging.getLogger("jqadaptor.fetcher")
        with mock.patch.object(log, "warning") as m:
            self.assertIsNone(await fetcher.get_security_list())
            self.assertTupleEqual(("not connected",), m.call_args.args)

    async def test_get_security_list(self):
        sec_list = await self.fetcher.get_security_list()
        print(sec_list[0])
        self.assertTrue(len(sec_list) > 0)

    async def test_get_bars(self):
        sec = "000001.XSHE"
        end = arrow.get("2020-04-04").date()
        frame_type = "1d"
        bars = await self.fetcher.get_bars(sec, end, 10, frame_type)
        self.assertEqual(bars[0]["frame"], arrow.get("2020-03-23").date())
        self.assertEqual(bars[-1]["frame"], arrow.get("2020-04-03").date())

        end = arrow.get("2020-04-03").date()
        frame_type = "1d"
        bars = await self.fetcher.get_bars(sec, end, 3, frame_type)
        self.assertEqual(bars[0]["frame"], arrow.get("2020-4-1").date())
        self.assertEqual(bars[-1]["frame"], end)

        end = arrow.get("2020-04-03 10:30:00", tzinfo="Asia/Shanghai").datetime
        frame_type = "30m"
        bars = await self.fetcher.get_bars(sec, end, 3, frame_type)
        if (
            bars[0]["frame"]
            != arrow.get("2020-04-02 15:00:00", tzinfo="Asia/Shanghai").datetime
        ):
            print(bars[0]["frame"])
            print(arrow.get("2020-04-02 15:00:00", tzinfo="Asia/Shanghai").datetime)

        self.assertEqual(
            bars[0]["frame"],
            arrow.get("2020-04-02 15:00:00", tzinfo="Asia/Shanghai").datetime,
        )
        self.assertEqual(bars[-1]["frame"], end)

        # 测试include_unclosed为False的情况
        import datetime

        end = datetime.date(2021, 2, 8)
        bars = await self.fetcher.get_bars(sec, end, 1, "1w", False)
        self.assertEqual(datetime.date(2021, 2, 5), bars["frame"][0])

        # 测试include_unclosed为True的情况
        sec = "000001.XSHE"
        end = datetime.date(2021, 2, 8)
        frame_type = "1w"
        bars = await self.fetcher.get_bars(sec, end, 1, frame_type, True)
        self.assertEqual(datetime.date(2021, 2, 8), bars["frame"][0])

    async def test_get_bars_with_exceptions(self):
        sec = "000001.XSHE"
        end = arrow.get("2020-04-04").date()
        frame_type = "1d"

        with mock.patch("jqdatasdk.get_bars", side_effect=[np.array([])]):
            bars = await self.fetcher.get_bars(sec, end, 3, frame_type)
            self.assertEqual(0, len(bars))

        with mock.patch("jqdatasdk.get_bars", side_effect=Exception("最大查询限制")):
            try:
                bars = await self.fetcher.get_bars(sec, end, 3, frame_type)
                self.assertTrue(False, "Expected FetcherQuotaError, got None")
            except FetcherQuotaError:
                self.assertTrue(True, "FetcherQuotaError throwed as expected")

    async def test_get_bars_not_in_trade(self):
        sec = "600891.XSHG"
        end = arrow.get("2020-03-05").date()
        bars = await self.fetcher.get_bars(sec, end, 7, "1d")
        print(bars)
        self.assertEqual(arrow.get("2020-2-21").date(), bars["frame"][0])
        self.assertAlmostEqual(1.25, bars[0]["open"], places=2)

        self.assertEqual(arrow.get("2020-02-26").date(), bars["frame"][3])
        self.assertAlmostEqual(1.18, bars["open"][3], places=2)

        self.assertEqual(arrow.get("2020-03-02").date(), bars["frame"][-1])
        self.assertAlmostEqual(1.13, bars["open"][-1], places=2)

        # 600721, ST百花， 2020-4-29停牌一天
        sec = "600721.XSHG"
        end = arrow.get("2020-04-30 10:30", tzinfo="Asia/Chongqing").datetime
        frame_type = "60m"

        bars = await self.fetcher.get_bars(sec, end, 6, frame_type)
        print(bars)
        self.assertEqual(6, len(bars))
        self.assertEqual(
            arrow.get("2020-04-27 15:00", tzinfo="Asia/Shanghai").datetime,
            bars["frame"][0],
        )
        # 检查是否停牌日被跳过
        self.assertEqual(
            arrow.get("2020-4-28 15:00", tzinfo="Asia/Shanghai"), bars["frame"][-2]
        )
        self.assertEqual(
            arrow.get("2020-04-30 10:30", tzinfo="Asia/Shanghai"), bars["frame"][-1]
        )
        self.assertAlmostEqual(5.47, bars["open"][0], places=2)
        self.assertAlmostEqual(5.26, bars["open"][-1], places=2)

        # 测试分钟级别未结束的frame能否获取
        end = arrow.get("2020-04-30 10:32", tzinfo="Asia/Shanghai").datetime
        bars = await self.fetcher.get_bars(sec, end, 7, "60m")
        print(bars)

        self.assertEqual(7, len(bars))
        self.assertEqual(
            arrow.get("2020-04-27 15:00", tzinfo="Asia/Shanghai"), bars["frame"][0]
        )
        self.assertEqual(
            arrow.get("2020-4-30 10:32", tzinfo="Asia/Shanghai"), bars["frame"][-1]
        )
        self.assertEqual(
            arrow.get("2020-04-30 10:30", tzinfo="Asia/Shanghai"), bars["frame"][-2]
        )

        self.assertAlmostEqual(5.47, bars["open"][0], places=2)
        self.assertAlmostEqual(5.26, bars["open"][-2], places=2)
        self.assertAlmostEqual(5.33, bars["open"][-1], places=2)

    async def test_get_valuation(self):
        sec = "000001.XSHE"
        day = arrow.get("2020-10-20").date()
        valuation = await self.fetcher.get_valuation(sec, day)
        self.assertSetEqual(set(valuation["code"].tolist()), set([sec]))

        sec = ["600000.XSHG", "000001.XSHE"]
        valuation = await self.fetcher.get_valuation(sec, day)
        self.assertSetEqual(set(valuation["code"].tolist()), set(sec))

        day = arrow.get("2020-11-2").date()
        vals = await self.fetcher.get_valuation(sec, day, 3)
        self.assertEqual(vals["frame"][0], arrow.get("2020-10-29").date())
        self.assertEqual(vals["frame"][-1], day)

        vals = await self.fetcher.get_valuation(None, day, 1)
        self.assertTrue(len(vals) > 1000)

    async def test_get_bars_batch(self):
        secs = ["000001.XSHE", "600000.XSHG"]
        end_at = arrow.get("2020-10-23").date()
        n_bars = 5
        frame_type = "1d"

        bars = await self.fetcher.get_bars_batch(secs, end_at, n_bars, frame_type)

        self.assertEqual(
            bars["000001.XSHE"]["frame"][0], arrow.get("2020-10-19").date()
        )
        self.assertEqual(
            bars["600000.XSHG"]["frame"][0], arrow.get("2020-10-19").date()
        )

    async def test_get_trade_price_limits(self):
        secs = ["000001.XSHE"]
        end_at = arrow.get("2020-10-23").date()
        bars = await self.fetcher.get_trade_price_limits(
            secs,
            dt=end_at,
        )
        self.assertIsNotNone(bars)
        try:
            await self.fetcher.get_trade_price_limits(123, dt=end_at)
        except Exception as e:
            self.assertIsInstance(e, TypeError)

        try:
            await self.fetcher.get_trade_price_limits(secs, dt=123)
        except Exception as e:
            self.assertIsInstance(e, TypeError)

        try:
            await self.fetcher.get_trade_price_limits(secs, dt=end_at)
        except Exception as e:
            self.assertIsInstance(e, TypeError)

        try:
            await self.fetcher.get_trade_price_limits(secs, dt=end_at)
        except Exception as e:
            self.assertIsInstance(e, ValueError)

        try:
            await self.fetcher.get_trade_price_limits(secs, dt=end_at)
        except Exception as e:
            self.assertIsInstance(e, TypeError)

    async def test_get_fund_list(self):
        fund_list = await self.fetcher.get_fund_list()
        self.assertTrue(len(fund_list) > 0)

        code = ["233333"]
        vals = await self.fetcher.get_fund_list(codes=code)
        self.assertFalse(len(vals))

    async def test_get_fund_portfolio_stock(self):

        await self.fetcher.get_fund_portfolio_stock(None, pub_date="2021-12-21")

        codes = ["000001"]
        portfolio_stocks = await self.fetcher.get_fund_portfolio_stock(codes)
        self.assertTrue(len(portfolio_stocks))

    async def test_get_fund_share_daily(self):
        code = ["512690"]
        day = arrow.get("2021-10-26").date()
        fund_share_daily = await self.fetcher.get_fund_share_daily(code, day=day)
        self.assertTrue(len(fund_share_daily))

        code = None
        day = arrow.get("2021-10-26").date()
        fund_share_daily = await self.fetcher.get_fund_share_daily(code, day=day)
        self.assertTrue(len(fund_share_daily))

    async def test_get_fund_net_value(self):
        codes = ["000029"]
        day = arrow.get("2021-10-26").date()
        fund_net_value = await self.fetcher.get_fund_net_value(codes, day=day)
        self.assertTrue(len(fund_net_value))

    async def test_get_quota(self):
        quota = await self.fetcher.get_quota()
        self.assertIsInstance(quota, dict)
        self.assertIn("total", quota)
        self.assertIn("spare", quota)

    def test_result_size_limit(self):
        self.assertEqual(self.fetcher.result_size_limit("bars"), 3000)

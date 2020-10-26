#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This is a awesome
        python script!"""
import logging
import os
import unittest

import arrow
from omicron.core.lang import async_run
from omicron.core.types import FrameType

import jqadaptor

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class TestJQ(unittest.TestCase):
    @async_run
    async def setUp(self) -> None:
        try:
            account = os.environ['jq_account']
            password = os.environ['jq_password']
            self.fetcher = await jqadaptor.create_instance(account=account,
                                                           password=password)
        except Exception as e:
            logger.exception(e)

    @async_run
    async def test_get_security_list(self):
        sec_list = await self.fetcher.get_security_list()
        print(sec_list[:5])

    @async_run
    async def test_get_bars(self):
        sec = '000001.XSHE'
        end = arrow.get('2020-04-04').date()
        frame_type = FrameType.DAY
        bars = await self.fetcher.get_bars(sec, end, 10, frame_type)
        print(bars)
        self.assertEqual(bars[0]['frame'], arrow.get('2020-03-23').date())
        self.assertEqual(bars[-1]['frame'], arrow.get('2020-04-03').date())

        end = arrow.get('2020-04-03').date()
        frame_type = FrameType.DAY
        bars = await self.fetcher.get_bars(sec, end, 3, frame_type)
        print(bars)
        self.assertEqual(bars[0]['frame'], arrow.get('2020-4-1').date())
        self.assertEqual(bars[-1]['frame'], end)

        end = arrow.get('2020-04-03 10:30:00', tzinfo='Asia/Shanghai').datetime
        frame_type = FrameType.MIN30
        bars = await self.fetcher.get_bars(sec, end, 3, frame_type)
        print(bars)
        self.assertEqual(bars[0]['frame'], arrow.get('2020-04-02 15:00:00', tzinfo='Asia/Shanghai').datetime)
        self.assertEqual(bars[-1]['frame'], end)

    @async_run
    async def test_get_bars_not_in_trade(self):
        sec = '600891.XSHG'
        end = arrow.get("2020-03-05").date()
        bars = await self.fetcher.get_bars(sec, end, 7, FrameType.DAY)
        print(bars)
        self.assertEqual(arrow.get('2020-2-21').date(), bars['frame'][0])
        self.assertAlmostEqual(1.25, bars[0]['open'])

        self.assertEqual(arrow.get('2020-02-26').date(), bars['frame'][3])
        self.assertAlmostEqual(1.18, bars['open'][3])

        self.assertEqual(arrow.get('2020-03-02').date(), bars['frame'][-1])
        self.assertAlmostEqual(1.13, bars['open'][-1])

        # 600721, ST百花， 2020-4-29停牌一天
        sec = '600721.XSHG'
        end = arrow.get('2020-04-30 10:30', tzinfo='Asia/Chongqing').datetime
        frame_type = FrameType.MIN60

        bars = await self.fetcher.get_bars(sec, end, 6, frame_type)
        print(bars)
        self.assertEqual(6, len(bars))
        self.assertEqual(arrow.get('2020-04-27 15:00', tzinfo='Asia/Shanghai'),
                         bars['frame'][0])
        # 检查是否停牌日被跳过
        self.assertEqual(arrow.get('2020-4-28 15:00', tzinfo='Asia/Shanghai'),
                         bars['frame'][-2])
        self.assertEqual(arrow.get('2020-04-30 10:30', tzinfo='Asia/Shanghai'),
                         bars['frame'][-1])
        self.assertAlmostEqual(5.47, bars['open'][0])
        self.assertAlmostEqual(5.26, bars['open'][-1])

        # 测试分钟级别未结束的frame能否获取
        end = arrow.get('2020-04-30 10:32', tzinfo='Asia/Shanghai').datetime
        bars = await self.fetcher.get_bars(sec, end, 7, FrameType.MIN60)
        print(bars)

        self.assertEqual(7, len(bars))
        self.assertEqual(arrow.get('2020-04-27 15:00', tzinfo='Asia/Shanghai'),
                         bars['frame'][0])
        self.assertEqual(arrow.get('2020-4-30 10:32', tzinfo='Asia/Shanghai'),
                         bars['frame'][-1])
        self.assertEqual(arrow.get('2020-04-30 10:30', tzinfo='Asia/Shanghai'),
                         bars['frame'][-2])

        self.assertAlmostEqual(5.47, bars['open'][0])
        self.assertAlmostEqual(5.26, bars['open'][-2])
        self.assertAlmostEqual(5.33, bars['open'][-1])

    @async_run
    async def test_get_valuation(self):
        sec = '000001.XSHE'
        day = arrow.get('2020-10-20').date()
        valuation = await self.fetcher.get_valuation(sec, day)
        self.assertSetEqual(set(valuation['code'].tolist()), set([sec]))


        sec = ['600000.XSHG', '000001.XSHE']
        valuation = await self.fetcher.get_valuation(sec, day)
        self.assertSetEqual(set(valuation['code'].tolist()), set(sec))

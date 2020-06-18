#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This is a awesome
        python script!"""
import logging
import os
import unittest

import arrow
import numpy
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
        end = arrow.get('2020-04-04')
        bars = await self.fetcher.get_bars(sec, end, 10, FrameType.DAY)
        import pprint
        pprint.pprint(bars)
        self.assertEqual(bars[0]['frame'], arrow.get('2020-03-23').date())
        self.assertEqual(bars[-1]['frame'], arrow.get('2020-04-03').date())

    @async_run
    async def test_get_bars_not_in_trade(self):
        sec = '600891.XSHG'
        end = arrow.get("2020-03-05")
        bars = await self.fetcher.get_bars(sec, end, 7, FrameType.DAY)
        print(bars)
        self.assertEqual(arrow.get('2020-2-26').date(), bars['frame'][0])
        self.assertAlmostEqual(1.18, bars[0]['open'])

        self.assertEqual(arrow.get('2020-03-02').date(), bars['frame'][3])
        self.assertAlmostEqual(1.13, bars['open'][3])

        self.assertEqual(arrow.get('2020-03-03').date(), bars['frame'][4])
        self.assertTrue(numpy.isnan(bars['open'][4]))

        self.assertEqual(arrow.get('2020-03-05').date(), bars['frame'][-1])
        self.assertTrue(numpy.isnan(bars['open'][-1]))

        # 600721, ST百花， 2020-4-29停牌一天
        sec = '600721.XSHG'
        end = arrow.get('2020-04-30 10:30', tzinfo='Asia/Chongqing')

        bars = await self.fetcher.get_bars(sec, end, 6, FrameType.MIN60)
        print(bars)
        self.assertEqual(6, len(bars))
        self.assertEqual(arrow.get('2020-04-28 15:00', tzinfo='Asia/Shanghai'),
                         bars['frame'][0])
        self.assertEqual(arrow.get('2020-04-30 10:30', tzinfo='Asia/Shanghai'),
                         bars['frame'][-1])
        self.assertAlmostEqual(5.37, bars['open'][0])
        self.assertAlmostEqual(5.26, bars['open'][-1])
        self.assertTrue(numpy.isnan(bars['open'][1]))
        
        sec = '600721.XSHG'
        end = arrow.get('2020-04-29 11:30', tzinfo='Asia/Chongqing')

        bars = await self.fetcher.get_bars(sec, end, 4, FrameType.MIN60)
        print(bars)
        self.assertEqual(4, len(bars))
        self.assertEqual(arrow.get('2020-04-28 14:00', tzinfo='Asia/Shanghai'),
                         bars['frame'][0])
        self.assertEqual(arrow.get('2020-04-29 11:30', tzinfo='Asia/Shanghai'),
                         bars['frame'][-1])
        self.assertAlmostEqual(5.37, bars['open'][0])
        self.assertTrue(numpy.isnan(bars['open'][-1]))



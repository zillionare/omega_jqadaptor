#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This is a awesome
        python script!"""
import logging
import os
import unittest

import arrow
from omicron.core import FrameType
from omicron.core.lang import async_run
from jqadaptor import create_instance

logger = logging.getLogger(__file__)


class TestJQ(unittest.TestCase):
    @async_run
    async def setUp(self) -> None:
        account = os.environ['jq_account']
        password = os.environ['jq_password']

        self.fetcher = await create_instance(account=account, password=password)

    @async_run
    async def test_get_security_list(self):
        sec_list = await self.fetcher.get_security_list()
        print(sec_list[:5])

    @async_run
    async def test_get_bars(self):
        sec = '000001.XSHE'
        end = arrow.get('2020-04-04')
        bars = await self.fetcher.get_bars(sec, end, 10, FrameType.DAY)
        self.assertEqual(bars[0]['frame'], arrow.get('2020-03-23').date())
        self.assertEqual(bars[-1]['frame'], arrow.get('2020-04-03').date())
        import pprint
        pprint.pprint(bars)

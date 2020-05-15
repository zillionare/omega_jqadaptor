"""Top-level package for SoloQuotes."""

__author__ = """Aaron Yang"""
__email__ = 'code@jieyu.ai'
__version__ = '0.1.0'

from .fetcher import Fetcher


async def create_instance(**kwargs):
    f = Fetcher()
    await f.create_instance(**kwargs)
    return f


__all__ = ['create_instance']

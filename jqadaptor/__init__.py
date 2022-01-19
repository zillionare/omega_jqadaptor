"""joinquant adaptor for zillionare"""

__author__ = """Aaron Yang"""
__email__ = "code@jieyu.ai"
__version__ = "1.0.6"

from .fetcher import Fetcher


async def create_instance(**kwargs):
    """create fetcher instance and start session
    Returns:
        [QuotesFetcher]: the fetcher
    """
    fetcher = Fetcher()
    await fetcher.create_instance(**kwargs)
    return fetcher


__all__ = ["create_instance"]

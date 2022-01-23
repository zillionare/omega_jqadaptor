"""joinquant adaptor for zillionare"""
import pkg_resources

from .fetcher import Fetcher

__author__ = """Aaron Yang"""
__email__ = "code@jieyu.ai"
__version__ = pkg_resources.get_distribution("zillionare-omega-adaptors-jq").version


async def create_instance(**kwargs):
    """create fetcher instance and start session
    Returns:
        [QuotesFetcher]: the fetcher
    """
    fetcher = Fetcher()
    await fetcher.create_instance(**kwargs)
    return fetcher


__all__ = ["create_instance"]

# History

## 0.1.0 (2020-04-10)

* First release on PyPI.

## 0.3.2 (2020-12-04)

* Alpha release of 0.3
* Features:
    1. get_bar
    2. get_security_list
    3. get_all_trade_days
    4. get_bars_batch
    5. get_valuation

## 1.0.0 (2021-3-29) (yanked)

due to severe bug found (github://#3), please don't use this version

bug fix:
    1. lock down jqdatasdk, sqlalchemy's version. Recently sqlalchemy's update (to 1.4) cause several incompatible issue.
    2. remove dependancy of omicron
    3. fix timezone issue of get_bars/get_bars_batch, see #2

# 1.0.1 (2021-3-30)

This is first official release of zillionare-omega-adaptors-jq.

* Features:
    1. get_bar
    2. get_security_list
    3. get_all_trade_days
    4. get_bars_batch
    5. get_valuation
 * bug fixes:
    github: #2, #3

# 1.0.2 (2021-3-30)

This is a patch just to add releae notes. It's identical to 1.0.1 on binary sense.
* change list
    1. add release notes

# 1.0.3 (2020-3-31)
* change list
    1. fetcher will not try fetching data after login failed. This is friendly to server.
    2. Fix: after use markdown as readme/history file type, forget correct manifest.in, this cause tox failed.

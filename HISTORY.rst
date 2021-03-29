=======
History
=======

0.1.0 (2020-04-10)
------------------

* First release on PyPI.

0.3.2 (2020-12-04)
-------------------
* Alpha release of 0.3
* Features:
    1. get_bar
    2. get_security_list
    3. get_all_trade_days
    4. get_bars_batch
    5. get_valuation

1.0.0 (2021-3-29)
-------------------
bug fix:
    1. lock down jqdatasdk, sqlalchemy's version. Recently sqlalchemy's update (to 1.4)
    cause several incompatible issue.
    2. remove dependancy of omicron
    3. fix timezone issue of get_bars/get_bars_batch, see #2

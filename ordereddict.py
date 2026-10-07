"""Compatibility shim.

The 2012 code ships a Python 2.4 backport of OrderedDict. Python 3.7+ has an
ordered dict built in and ``collections.OrderedDict`` has been available since
2.7, so the backport is simply re-exported here for ``from ordereddict import
OrderedDict`` in layer.py.
"""

from collections import OrderedDict  # noqa: F401

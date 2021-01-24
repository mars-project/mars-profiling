"""Main module of pandas-profiling.

.. include:: ../../README.md
"""

from mars_profiling.config import Config, config
from mars_profiling.controller import pandas_decorator
from mars_profiling.profile_report import ProfileReport
from mars_profiling.version import __version__

clear_config = ProfileReport.clear_config

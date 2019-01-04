from .utils import *
from .namespace import *
from .typing import Col, pandas_wrap

__all__ = ['patch_spark', 'read_csv', 'Col', 'pandas_wrap']

def _auto_patch():
    import os
    import logging
    # Autopatching is on by default.
    x = os.getenv("SPARK_PANDAS_AUTOPATCH", "true")
    if x.lower() in ("true", "1", "enabled"):
        logger = logging.getLogger('spark')
        logger.info("Patching spark automatically. You can disable it by setting SPARK_PANDAS_AUTOPATCH=false in your environment")
        patch_spark()

_auto_patch()

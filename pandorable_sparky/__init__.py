#
# Copyright (C) 2019 Databricks, Inc.
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
        logger.info("Patching spark automatically. You can disable it by setting "
                    "SPARK_PANDAS_AUTOPATCH=false in your environment")
        patch_spark()

_auto_patch()

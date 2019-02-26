#!/usr/bin/env bash

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

# Runs both doctests and unit tests by default, otherwise hands arguments over to nose.

# The current directory of the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/env_setup.sh

FWDIR="$( cd "$DIR"/.. && pwd )"
cd "$FWDIR"

if [ "$#" = 0 ]; then
    ARGS="--nologcapture --all-modules --verbose --with-doctest"
else
    ARGS="$@"
fi
exec nosetests $ARGS --where "$FWDIR"

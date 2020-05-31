#!/usr/bin/env bash

#
# Copyright (C) 2020 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

echo "Downloading Spark if necessary"
echo "Spark version = $SPARK_VERSION"

sparkVersionsDir="$HOME/.cache/spark-versions"
mkdir -p "$sparkVersionsDir"
sparkFile="spark-$SPARK_VERSION-bin-hadoop2.7"
sparkBuild="spark-$SPARK_VERSION-rc$SPARK_RC_VERSION-bin-hadoop2.7"
sparkBuildDir="$sparkVersionsDir/$sparkBuild"

if [[ -d "$sparkBuildDir" ]]; then
    echo "Skipping download - found Spark dir $sparkBuildDir"
else
    echo "Missing $sparkBuildDir, downloading archive"

    # If not already found,
    if ! [[ -d "$sparkBuildDir" ]]; then
        sparkURL="https://dist.apache.org/repos/dist/dev/spark/v$SPARK_VERSION-rc$SPARK_RC_VERSION-bin/$sparkFile.tgz"
        echo "Downloading $sparkURL ..."
        # Test whether it's reachable
        if curl -s -I -f -o /dev/null "$sparkURL"; then
            curl -s "$sparkURL" | tar xz --directory "$sparkVersionsDir"
            mv "$sparkVersionsDir/$sparkFile" "$sparkVersionsDir/$sparkBuild"
        else
            echo "Could not reach $sparkURL"
        fi
    fi

    echo "Content of $sparkBuildDir:"
    ls -la "$sparkBuildDir"
fi

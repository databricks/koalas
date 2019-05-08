#!/usr/bin/env bash

#
# Copyright (C) 2019 Databricks, Inc.
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
sparkBuild="spark-$SPARK_VERSION-bin-hadoop2.7"
sparkBuildDir="$sparkVersionsDir/$sparkBuild"

if [[ -d "$sparkBuildDir" ]]; then
    echo "Skipping download - found Spark dir $sparkBuildDir"
else
    echo "Missing $sparkBuildDir, downloading archive"

    # Get a local ASF mirror, as HTTPS
    function apache_mirror() {
        local mirror_url=$(curl -s https://www.apache.org/dyn/closer.lua?preferred=true)
        echo "${mirror_url/http:/https:}"
    }
    # Try mirrors then fall back to dist.apache.org
    for mirror in $(apache_mirror) $(apache_mirror) $(apache_mirror) https://dist.apache.org/repos/dist/release/; do
        # If not already found,
        if ! [[ -d "$sparkBuildDir" ]]; then
            sparkURL="$mirror/spark/spark-$SPARK_VERSION/$sparkBuild.tgz"
            echo "Downloading $sparkURL ..."
            # Test whether it's reachable
            if curl -s -I -f -o /dev/null "$sparkURL"; then
                curl -s "$sparkURL" | tar xz --directory "$sparkVersionsDir"
            else
                echo "Could not reach $sparkURL"
            fi
        fi
    done

    echo "Content of $sparkBuildDir:"
    ls -la "$sparkBuildDir"
fi


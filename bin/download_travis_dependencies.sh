#!/usr/bin/env bash

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


#!/usr/bin/env bash
# Sets up enviornment from test. Should be sourced in other scripts.

if [ -z "$SPARK_HOME" ]; then
    echo 'You need to set $SPARK_HOME to run these tests.' >&2
    exit 1
fi

LIBS=""
for lib in "$SPARK_HOME/python/lib"/*zip ; do
  LIBS=$LIBS:$lib
done

export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python:$LIBS:.

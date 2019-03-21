[![Build Status](https://travis-ci.com/databricks/spark-pandas.svg?token=Rzzgd1itxsPZRuhKGnhD&branch=master)](https://travis-ci.com/databricks/spark-pandas)

# Koala: Pandas APIs on Apache Spark

This package modifies PySpark's dataframe API to 
make it compliant (mostly) with the Pandas API.

Requirements:
 - Spark 2.4. Some older versions of Spark may work too but they are not supported.
 - a recent version of Pandas. It is officially developed against 0.23+ but some other versions may work too.
 - Python 3.5+ if you want to use type hints in UDFs. Work is ongoing to also support Python 2.

How to use:

```py
import databricks.koala
```

That's it. Now you have turned all the Spark Dataframes 
that will be created from now on into compliant Pandas 
dataframes.

Example:

```py
import pandas as pd
pdf = pd.DataFrame({'x':range(3), 'y':['a','b','b'],'z':['a','b','b']})

df = spark.createDataFrame(pdf)
# Rename the columns
df.columns = ['x', 'y', 'z1']


# Do some operations in place:
df['x2'] = df.x * df.x
```

## How to use

Pending publication on the PyPI repository, a compiled package can be installed by using
this URL:

```bash
pip install https://s3-us-west-2.amazonaws.com/databricks-tjhunter/pandorable_sparky/pandorable_sparky-0.0.5-py3-none-any.whl
```

## What is available

Pandas has a very extensive API and currently Koala only supports a subset of the Pandas API.
The authoritative status is this spreadsheet, available as a Google document:

https://docs.google.com/spreadsheets/d/1GwBvGsqZAFFAD5PPc_lffDEXith353E1Y7UV6ZAAHWA/edit?usp=sharing


## Current status

This project is a technology preview that is meant to show the current available that is
available. In this preview, you should expect the following differences:

 - some functions may be missing (see the [What is available](#what-is-available) section)

 - some behaviour may be different, in particular in the treatment of nulls: Pandas uses
   Not a Number (NaN) special constants to indicate missing values, while Spark has a
   special flag on each value to indicate missing values. We would love to hear your use
   case if you find differences.
   
 - because Spark is lazy in nature, some operations like creating new columns only get 
   performed when Spark needs to print or write the dataframe.



## How to contribute

Are you missing a function from Pandas? No problem! Most functions are very easy to add
by simply wrapping the existing Pandas function.

 1. Look at [the list of implemented functions](https://docs.google.com/spreadsheets/d/1GwBvGsqZAFFAD5PPc_lffDEXith353E1Y7UV6ZAAHWA/edit?usp=sharing) to see if a pull request already exists
 
 2. Wrap your function and submit it as a pull request
 
If the function already has the same name in Apache Spark and if the results differ, the 
general policy is to follow the Behaviour of Spark and to document the changes.
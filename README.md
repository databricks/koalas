![](https://api.travis-ci.com/databricks/spark-pandas.svg)

# spark-pandas
Spark is cute. Sparky is _pandorable_.

Pandas dataframe APIs on Apache Spark

This package modifies PySpark's dataframe API to 
make it compliant (mostly) with the Pandas API (i.e. 
more pandorable).

Requirements:
 - spark 2.4 (at least 2.3+ if you want to use UDFs)
 - a recent version of pandas. It is officially developped against 0.23+ but some other versions may work too.
 - python 3.5+ if you want to use type hints in UDFs.

How to use:

```py
import pandorable_spark
```

That's it. Now you have turned all the Spark Dataframes 
that will be created from now on into compliant pandas 
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
 



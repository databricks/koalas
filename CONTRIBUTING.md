# Contributing to Koalas

We welcome community contributions to Koalas. 

## How Can I Contribute?

### Filing Issues

If you notice a bug in the code or documentation, or have suggestions for how we can improve either,
 feel free to create an issue on the GitHub "issues" tab using GitHub's "issue" form.
 Before submitting an issue, ensure the issue was not already reported by searching on GitHub under Issues [https://github.com/databricks/spark-pandas/issues].
 
1. If the function from Pandas is missing, most functions are very easy to add by simply wrapping the existing Pandas function. Please look at [the list of implemented functions](https://docs.google.com/spreadsheets/d/1GwBvGsqZAFFAD5PPc_lffDEXith353E1Y7UV6ZAAHWA/edit?usp=sharing) to see if a pull request already exists
 
2. If the function already has the same name in Apache Spark and if the results differ, the general policy is to follow the behaviour of Spark and to document the changes.

### Local development and testing

#### Prerequisites
We recommend installing Koalas in its own conda environment for development, as follows:

```bash
conda create --name koalas-dev-env python=3.6
source activate koalas-dev-env
conda install -c conda-forge pyspark=2.4 pandas pyarrow=0.10 decorator flake8 nose
cd /path/to/spark-pandas
pip install -e .  # installs koalas from current checkout
```

#### Run unit tests

Run all the unit test cases: 
```bash
./dev/run-tests.sh
```

Run the unit test cases in a specific test file:
```bash
python databricks/koalas/tests/test_dataframe.py
```

Run a specific unit test case:
```bash
python databricks/koalas/tests/test_dataframe.py DataFrameTest.test_Dataframe
```

### Contributing Code 

Fork the Github repository at https://github.com/databricks/spark-pandas if you haven’t already. Please send a GitHub Pull Request to opengovernment with a clear list of what you've done

(read more about [pull requests](https://help.github.com/en/articles/about-pull-requests)).


### Coding conventions

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with one exception: lines can be up to 100 characters in length, not 79.

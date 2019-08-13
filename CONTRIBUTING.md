# Contributing Guide <!-- omit in toc -->

This document gives guidance to developers if they plan to contribute to Koalas.

- [Types of Contributions](#types-of-contributions)
- [Step-by-step Guide](#step-by-step-guide)
- [Environment Setup](#environment-setup)
- [Working with Git](#working-with-git)
- [Running Tests](#running-tests)
- [Building Documentation](#building-documentation)
- [Coding Conventions](#coding-conventions)
- [Release Instructions](#release-instructions)

## Types of Contributions

The largest amount of work consists simply of implementing the pandas API using Spark's built-in functions, which is usually straightforward. But there are many different forms of contributions in addition to writing code:

1. Use the project and provide feedback, by creating new tickets or commenting on existing relevant tickets.

2. Review existing pull requests.

3. Improve the project's documentation.

4. Write blog posts or tutorial articles evangelizing Koalas and help new users learn Koalas.

5. Give a talk about Koalas at your local meetup or a conference.


## Step-by-step Guide For Code Contributions

1. Read and understand the [Design Principles](https://github.com/databricks/koalas/blob/master/README.md#design-principles) for the project. Contributions should follow these principles.

2. Signaling your work: If you are working on something, comment on the relevant ticket that you are doing so to avoid multiple people taking on the same work at the same time. It is also a good practice to signal that your work has stalled or you have moved on and want somebody else to take over.

3. Understand what the functionality is in pandas or in Spark.

4. Implement the functionality, with test cases providing close to 100% statement coverage. Document the functionality.

5. Run existing and new test cases to make sure they still pass. Also run the linter `dev/lint-python`.

6. Build the docs (`make html` in `docs` directory) and verify the docs related to your change look OK.

7. Submit a pull request, and be responsive to code review feedback from other community members.

That's it. Your contribution, once merged, will be available in the next release.


## Environment Setup

**Conda**

We recommend setting up a Conda environment for development:
```bash
# Python 3.6+ is required
conda create --name koalas-dev-env python=3.6
conda activate koalas-dev-env
conda install -c conda-forge pyspark=2.4
conda install -c conda-forge --yes --file requirements-dev.txt
pip install -e .  # installs koalas from current checkout
```

Once setup, make sure you switch to `koalas-dev-env` before development:
```bash
conda activate koalas-dev-env
```

**pip**

You can use `pip` alternatively if your Python is 3.5+.
```bash
pip install pyspark=2.4
pip install -r requirements-dev.txt
pip install -e .  # installs koalas from current checkout
```

## Working with Git

[The instruction on Github](https://help.github.com/set-up-git-redirect) has all steps you need to take to install git, set up
your SSH key and fork and clone to your local repository.

Your master branch should only reflect production-ready code, therefore, in order to start contributing to the project, you 
need to create a feature branch for your changes. For example, you could do:

```bash
git checkout -b your-new-pr
```

And it will direct your local directory to the specified new branch. And it is better to keep your master branch up to date with 
the latest upstream master version. Add all the changes in this feature branch and make sure all tests are passed. You can 
then do:

```bash
git add changed-files
git commit -m 'Brief description'
git push --set-upstream origin your-new-pr
```

After this, you could go to Github koalas repository and create the PR.

## Running Tests

There is a script `./dev/pytest` which is exactly same as `pytest` but with some default settings to run Koalas tests easily.

To run all the tests, similar to our CI pipeline:
```bash
# Run all unittest and doctest
./dev/pytest
```

To run a specific test file:
```bash
# Run unittest
./dev/pytest -k test_dataframe.py

# Run doctest
./dev/pytest -k series.py --doctest-modules databricks
```

To run a specific doctest/unittest:
```bash
# Run unittest
./dev/pytest -k "DataFrameTest and test_Dataframe"

# Run doctest
./dev/pytest -k DataFrame.corr --doctest-modules databricks
```

Note that `-k` is used for simplicity although it takes an expression. You can use `--verbose` to check what to filter. See `pytest --help` for more details.


## Building Documentation

To build documentation via Sphinx:

```bash
cd docs && make clean html
```

It generates HTMLs under `docs/build/html` directory. Open `docs/build/html/index.html` to check if documentation is built properly.


## Coding Conventions
We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with one exception: lines can be up to 100 characters in length, not 79.

## Doctest Conventions

When writing doctests, usually the doctests in pandas are converted into Koalas to make sure the same codes work in Koalas.
In general, doctests should be grouped logically by seperating a newline.

For instance, the first block is for the statements for preparation, the second block is for using the function with a specific argument,
and third block is for another argument. As a example, please refer [DataFrame.rsub](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rsub.html#pandas.DataFrame.rsub) in pandas.

These blocks should be consistently separated in Koalas, and more doctests should be added if the coverage of the doctests or the number of examples to show is not enough even though they are different from pandas'.

## Release Instructions
Only project maintainers can do the following.

Step 1. Make sure version is set correctly in `databricks/koalas/version.py`.

Step 2. Make sure the build is green.

Step 3. Create a new release on GitHub. Tag it as the same version as the setup.py.
If the version is "0.1.0", tag the commit as "v0.1.0".

Step 4. Upload the package to PyPi:
```bash
rm -rf dist/koalas*
python setup.py sdist bdist_wheel
export package_version=$(python setup.py --version)
echo $package_version

python3 -m pip install --user --upgrade twine

# for test
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/koalas-$package_version-py3-none-any.whl dist/koalas-$package_version.tar.gz

# for release
python3 -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/koalas-$package_version-py3-none-any.whl dist/koalas-$package_version.tar.gz
```

Step 5. Verify the uploaded package can be installed and executed.
One unofficial tip is to run the doctests of Koalas within a Python interpretor after installing it.

```python
import os

from pytest import main
import databricks

test_path = os.path.abspath(os.path.dirname(databricks.__file__))
main(['-k', '-to_delta -read_delta', '--verbose', '--showlocals', '--doctest-modules', test_path])
```

Note that this way might require additional settings, for instance, environment variables.


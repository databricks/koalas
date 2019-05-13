# Contributing Guide <!-- omit in toc -->

This document gives guidance to developers if they plan to contribute to Koalas. The largest amount of work consists simply in implementing the pandas API in Spark terms, which is usually straightforward.

- [Step-by-step Guide](#step-by-step-guide)
- [Environment Setup](#environment-setup)
- [Running Tests](#running-tests)
- [Building Documentation](#building-documentation)
- [Coding Conventions](#coding-conventions)
- [Release Instructions](#release-instructions)

## Step-by-step Guide

1. Read and understand the [Design Principles](https://github.com/databricks/koalas/blob/master/README.md#design-principles) for the project. Contributions should match these projects.

2. Signaling your work: If you are working on something, comment on the relevant ticket that you are doing so to avoid multiple people taking on the same work at the same time. It is also a good practice to signal that your work has stalled or you have moved on and want somebody else to take over.

3. Understand what the functionality is in pandas or in Spark.

4. Implement the functionality, with test cases providing close to 100% statement coverage. Document the functionality.

5. Run existing and new test cases to make sure they still pass. Also run the linter `dev/lint-python`.

6. Submit a pull request, and be responsive to code review feedback from other community members.

That's it. Your contribution, once merged, will be available in the next release.


## Environment Setup

We recommend setting up a Conda environment for development:
```bash
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

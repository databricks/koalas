from setuptools import setup

setup(
    name='pandorable_sparky',
    version='0.0.5',
    packages=['pandorable_sparky', 'pandorable_sparky._dask_stubs'],
    install_requires=[
        'pyspark>=2.4.0',
        'pandas>=0.23',
        'decorator',
        'pyarrow>=0.10,<0.11'],  # See discussion on https://github.com/databricks/spark-pandas/issues/26
    author="Timothy Hunter",
    author_email="tim@databricks.com",
    license='Apache 2.0',
    long_description=open('README.md').read(),
)

from setuptools import setup

setup(
    name='pandorable_sparky',
    version='0.0.1',
    packages=['pandorable_sparky', 'pandorable_sparky._dask_stubs'],
    install_requires=['pyspark', 'pandas', 'decorator'],
    author="Timothy Hunter",
    author_email="tim@databricks.com",
    license='Apache 2.0',
    long_description=open('README.md').read(),
)
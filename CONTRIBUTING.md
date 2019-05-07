# Contributing Guide

This document gives guidance to developers if they plan to contribute to Koalas.

The largest amount of work consists simply in implementing the pandas API in Spark terms, which is usually straightforward. Because this project is aimed at users who may not be familiar with the intimate technical details of pandas or Spark, a few points have to be respected:

- *Signaling your work*: If you are working on something, comment on the relevant ticket that are you doing so to avoid multiple people taking on the same work at the same time. It is also a good practice to signal that your work has stalled or you have moved on and want somebody else to take over.

- *Testing*: For pandas functions, the testing coverage should be as good as in pandas. This is easily done by copying the relevant tests from pandas or dask into Koalas.

- *Documentation*: For the implemented parameters, the documentation should be as comprehensive as in the corresponding parameter in PySpark or pandas. A recommended way to add documentation is to start with the docstring of the corresponding function in PySpark or pandas, and adapt it for Koalas. If you are adding a new function, also add it to the API reference doc index page in `docs/source/reference` directory.

- *Exposing details*: Do not add internal fields if possible. Nearly all the state should be already encapsulated in the Spark dataframe. Similarly, do not replicate the abstract interfaces found in Pandas. They are meant to be a protocol to exchange data at high performance with numpy, which we cannot do anyway.

- *Monkey patching, field introspection and other advanced python techniques*: The current design should not require any of these. Avoid using these advanced techniques as much as possible.

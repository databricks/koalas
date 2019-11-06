======
Window
======
.. currentmodule:: databricks.koalas.window

Rolling objects are returned by ``.rolling`` calls: :func:`koalas.DataFrame.rolling`, :func:`koalas.Series.rolling`, etc.
Expanding objects are returned by ``.expanding`` calls: :func:`koalas.DataFrame.expanding`, :func:`koalas.Series.expanding`, etc.

Standard moving window functions
--------------------------------

.. autosummary::
   :toctree: api/

   Rolling.count
   Rolling.sum
   Rolling.min
   Rolling.max
   Rolling.mean

Standard expanding window functions
-----------------------------------

.. autosummary::
   :toctree: api/

   Expanding.count
   Expanding.sum
   Expanding.min
   Expanding.max
   Expanding.mean

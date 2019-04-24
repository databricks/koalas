
if __name__ == "__main__":
    import doctest
    import databricks.koalas as ks
    from databricks.koalas import frame, series
    doctest.testmod(frame, extraglobs={"ks": ks})
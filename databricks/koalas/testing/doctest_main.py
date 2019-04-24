

if __name__ == "__main__":
    import doctest
    import databricks.koalas as ks
    from databricks.koalas import frame, series, namespace
    modules = [frame, series, namespace]
    for m in modules:
        doctest.testmod(m, extraglobs={"ks": ks})